from torch import Tensor
import torch
from unittest.mock import patch
from typing import Dict
import math
from comfy.ldm.wan.model import sinusoidal_embedding_1d
from contextlib import ExitStack

def create_wanvideo_patches_dict(new_model, type="lite"):
    diffusion_model = new_model.get_model_object("diffusion_model")
    
    class ForwardPatcher:
        def __enter__(self):
            self.stack = ExitStack()
            
            if type == "lite":
                # patch Taylor 的 forward 函数
                self.stack.enter_context(patch.object(
                    diffusion_model,
                    'forward_orig',
                    taylorseer_lite_wanvideo_forward.__get__(diffusion_model, diffusion_model.__class__)
                ))
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.stack.close()
    
    return ForwardPatcher()

def taylorseer_lite_wanvideo_forward(
    self,
    x,
    t,
    context,
    clip_fea=None,
    freqs=None,
    transformer_options={},
    **kwargs,
):
    
    # embeddings
    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)

    # time embeddings
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
    e = e.reshape(t.shape[0], -1, e.shape[-1])
    e0 = self.time_projection(e).unflatten(2, (6, self.dim))

    full_ref = None
    if self.ref_conv is not None:
        full_ref = kwargs.get("reference_latent", None)
        if full_ref is not None:
            full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
            x = torch.concat((full_ref, x), dim=1)

    # context
    context = self.text_embedding(context)

    context_img_len = None
    if clip_fea is not None:
        if self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
        context_img_len = clip_fea.shape[-2]

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    cal_type(cache_dic=self.cache_dic, current=self.current)
    self.current['stream'] = 'final_stream'
    self.current['layer'] = 0
    self.current['module'] = 'final'
    if self.current['type'] == 'full':
        for i, block in enumerate(self.blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len)
                    return out
                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)

        # head
        x = self.head(x, e)

        if full_ref is not None:
            x = x[:, full_ref.shape[1]:]

        derivative_approximation(cache_dic=self.cache_dic, current=self.current, feature=x)
    elif self.current['type'] == 'Taylor':
        x = taylor_formula(cache_dic=self.cache_dic, current=self.current)
    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    self.current["step"] += 1
    return x

def cache_init_wanvideo(fresh_threshold, max_order, first_enhance, last_enhance, steps):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1]={}
    cache_index[-1]={}
    cache_index['layer_index']={}

    cache[-1]['final_stream'] = {}
    cache_dic['cache_counter'] = 0

    final_stream_modules = ['final']

    cache[-1]['final_stream'][0] = {}
    for module in final_stream_modules:
        cache[-1]['final_stream'][0][module] = {}
    cache_index[-1][0] = {}

    cache_dic['taylor_cache'] = False
    cache_dic['Delta-DiT'] = False
    
    cache_dic['cache_type'] = 'random'
    cache_dic['cache_index'] = cache_index
    cache_dic['cache'] = cache
    cache_dic['fresh_ratio_schedule'] = 'ToCa' 
    cache_dic['fresh_ratio'] = 0.0
    cache_dic['fresh_threshold'] = fresh_threshold
    cache_dic['force_fresh'] = 'global' 
    cache_dic['soft_fresh_weight'] = 0.0
    cache_dic['taylor_cache'] = True
    cache_dic['max_order'] = max_order
    cache_dic['first_enhance'] = first_enhance
    cache_dic['last_enhance'] = last_enhance
    cache_dic['cal_threshold'] = fresh_threshold

    current = {}
    current['current_activated_step'] = 0
    current['previous_activated_step'] = 0
    current['step'] = 0
    current['num_steps'] = steps
    current['type'] = None
    current['layer'] = None
    current['stream'] = None
    current['module'] = None

    return cache_dic, current

def force_scheduler(cache_dic, current):
    if cache_dic['fresh_ratio'] == 0:
        # FORA
        linear_step_weight = 0.0
    else: 
        # TokenCache
        linear_step_weight = 0.0
    step_factor = torch.tensor(1 - linear_step_weight + 2 * linear_step_weight * current['step'] / current['num_steps'])
    threshold = torch.round(cache_dic['fresh_threshold'] / step_factor)

    # no force constrain for sensitive steps, cause the performance is good enough.
    # you may have a try.
    
    cache_dic['cal_threshold'] = threshold

def cal_type(cache_dic, current):
    '''
    Determine calculation type for this step
    '''
    if (cache_dic['fresh_ratio'] == 0.0) and (not cache_dic['taylor_cache']):
        # FORA:Uniform
        first_step = (current['step'] == 0)
    else:
        # ToCa: First enhanced
        first_step = (current['step'] < cache_dic['first_enhance'])
        #first_step = (current['step'] <= 3)

    if not first_step:
        fresh_interval = cache_dic['cal_threshold']
    else:
        fresh_interval = cache_dic['fresh_threshold']

    if (first_step) or (cache_dic['cache_counter'] == fresh_interval - 1 ) or (current['step'] >= cache_dic['last_enhance'] - 1):
        current['type'] = 'full'
        cache_dic['cache_counter'] = 0
        current['previous_activated_step'] = current['current_activated_step']
        current['current_activated_step'] = current['step']
        force_scheduler(cache_dic, current)
    
    elif (cache_dic['taylor_cache']):
        cache_dic['cache_counter'] += 1
        current['type'] = 'Taylor'
        

    elif (cache_dic['cache_counter'] % 2 == 1): # 0: ToCa-Aggresive-ToCa, 1: Aggresive-ToCa-Aggresive
        cache_dic['cache_counter'] += 1
        current['type'] = 'ToCa'
    # 'cache_noise' 'ToCa' 'FORA'
    elif cache_dic['Delta-DiT']:
        cache_dic['cache_counter'] += 1
        current['type'] = 'Delta-Cache'
    else:
        cache_dic['cache_counter'] += 1
        current['type'] = 'ToCa'
        #if current['step'] < 25:
        #    current['type'] = 'FORA'
        #else:    
        #    current['type'] = 'aggressive'
######################################################################
    #if (current['step'] in [3,2,1,0]):
    #    current['type'] = 'full'

@torch._dynamo.disable
def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    difference_distance = current['current_activated_step'] - current['previous_activated_step']
    #difference_distance = current['activated_times'][-1] - current['activated_times'][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic['max_order']):
        if (cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i]) / difference_distance
        else:
            break

    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors

@torch._dynamo.disable
def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    x = current['step'] - current['current_activated_step']
    output = 0
    
    for i in range(len(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']])):
        output += (1 / math.factorial(i)) * cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i] * (x ** i)
    
    return output