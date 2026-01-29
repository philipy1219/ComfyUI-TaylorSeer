import torch
from unittest.mock import patch
from typing import Dict
import math
from contextlib import ExitStack
import comfy.ldm.common_dit

def create_lumina_patches_dict(new_model, type="lite"):
    diffusion_model = new_model.get_model_object("diffusion_model")
    
    class ForwardPatcher:
        def __enter__(self):
            self.stack = ExitStack()
            
            if type == "lite":
                # patch Taylor 的 forward 函数
                self.stack.enter_context(patch.object(
                    diffusion_model,
                    '_forward',
                    taylorseer_lite_lumina_forward.__get__(diffusion_model, diffusion_model.__class__)
                ))
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.stack.close()
    
    return ForwardPatcher()

def taylorseer_lite_lumina_forward(
    self,
    x,
    timesteps,
    context,
    num_tokens, 
    attention_mask=None, 
    ref_latents=[], 
    ref_contexts=[], 
    siglip_feats=[],
    transformer_options={}, 
    **kwargs
):
    omni = len(ref_latents) > 0
    if omni:
        timesteps = torch.cat([timesteps * 0, timesteps], dim=0)
    t = 1.0 - timesteps
    cap_feats = context
    cap_mask = attention_mask
    bs, c, h, w = x.shape
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
    """
    Forward pass of NextDiT.
    t: (N,) tensor of diffusion timesteps
    y: (N,) tensor of text tokens/features
    """

    t = self.t_embedder(t * self.time_scale, dtype=x.dtype)  # (N, D)
    adaln_input = t

    if self.clip_text_pooled_proj is not None:
        pooled = kwargs.get("clip_text_pooled", None)
        if pooled is not None:
            pooled = self.clip_text_pooled_proj(pooled)
        else:
            pooled = torch.zeros((x.shape[0], self.clip_text_dim), device=x.device, dtype=x.dtype)

        adaln_input = self.time_text_embed(torch.cat((t, pooled), dim=-1))

    x_is_tensor = isinstance(x, torch.Tensor)
    img, mask, img_size, cap_size, freqs_cis, timestep_zero_index = self.patchify_and_embed(x, cap_feats, cap_mask, adaln_input, num_tokens, ref_latents=ref_latents, ref_contexts=ref_contexts, siglip_feats=siglip_feats, transformer_options=transformer_options)
    freqs_cis = freqs_cis.to(img.device)

    patches = transformer_options.get("patches", {})
    cond_or_uncond = transformer_options.get("cond_or_uncond", [])
    if cond_or_uncond == [1]:
        cache_dic_now = self.cache_dic_negative
        current_now = self.current_negative
    else:
        cache_dic_now = self.cache_dic
        current_now = self.current

    cal_type(cache_dic=cache_dic_now, current=current_now)
    current_now['stream'] = 'final_stream'
    current_now['layer'] = 0
    current_now['module'] = 'final'
    transformer_options["total_blocks"] = len(self.layers)
    transformer_options["block_type"] = "double"
    if current_now['type'] == 'full':
        img_input = img
        for i, layer in enumerate(self.layers):
            transformer_options["block_index"] = i
            img = layer(img, mask, freqs_cis, adaln_input, timestep_zero_index=timestep_zero_index, transformer_options=transformer_options)
            if "double_block" in patches:
                for p in patches["double_block"]:
                    out = p({"img": img[:, cap_size[0]:], "img_input": img_input[:, cap_size[0]:], "txt": img[:, :cap_size[0]], "pe": freqs_cis[:, cap_size[0]:], "vec": adaln_input, "x": x, "block_index": i, "transformer_options": transformer_options})
                    if "img" in out:
                        img[:, cap_size[0]:] = out["img"]
                    if "txt" in out:
                        img[:, :cap_size[0]] = out["txt"]

        img = self.final_layer(img, adaln_input, timestep_zero_index=timestep_zero_index)
        derivative_approximation(cache_dic=cache_dic_now, current=current_now, feature=img)
    elif current_now['type'] == 'Taylor':
        img = taylor_formula(cache_dic=cache_dic_now, current=current_now)
    img = self.unpatchify(img, img_size, cap_size, return_tensor=x_is_tensor)[:, :, :h, :w]
    current_now["step"] += 1
    return -img

def cache_init_lumina(fresh_threshold, max_order, first_enhance, last_enhance, steps):   

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