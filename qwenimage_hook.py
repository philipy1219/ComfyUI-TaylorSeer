from torch import Tensor
import torch
from unittest.mock import patch
from typing import Dict
import math
from comfy.ldm.wan.model import sinusoidal_embedding_1d
from contextlib import ExitStack
import copy

def create_qwenimage_patches_dict(new_model, type="lite"):
    diffusion_model = new_model.get_model_object("diffusion_model")
    
    class ForwardPatcher:
        def __enter__(self):
            self.stack = ExitStack()
            
            if type == "lite":
                # patch Taylor 的 forward 函数
                self.stack.enter_context(patch.object(
                    diffusion_model,
                    '_forward',
                    taylorseer_lite_qwenimage_forward.__get__(diffusion_model, diffusion_model.__class__)
                ))
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.stack.close()
    
    return ForwardPatcher()

def taylorseer_lite_qwenimage_forward(
    self,
    x,
    timesteps,
    context,
    attention_mask=None,
    guidance: torch.Tensor = None,
    ref_latents=None,
    transformer_options={},
    control=None,
    **kwargs
):
    
    timestep = timesteps
    encoder_hidden_states = context
    encoder_hidden_states_mask = attention_mask

    hidden_states, img_ids, orig_shape = self.process_img(x)
    num_embeds = hidden_states.shape[1]

    if ref_latents is not None:
        h = 0
        w = 0
        index = 0
        index_ref_method = kwargs.get("ref_latents_method", "index") == "index"
        for ref in ref_latents:
            if index_ref_method:
                index += 1
                h_offset = 0
                w_offset = 0
            else:
                index = 1
                h_offset = 0
                w_offset = 0
                if ref.shape[-2] + h > ref.shape[-1] + w:
                    w_offset = w
                else:
                    h_offset = h
                h = max(h, ref.shape[-2] + h_offset)
                w = max(w, ref.shape[-1] + w_offset)

            kontext, kontext_ids, _ = self.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset)
            hidden_states = torch.cat([hidden_states, kontext], dim=1)
            img_ids = torch.cat([img_ids, kontext_ids], dim=1)

    txt_start = round(max(((x.shape[-1] + (self.patch_size // 2)) // self.patch_size) // 2, ((x.shape[-2] + (self.patch_size // 2)) // self.patch_size) // 2))
    txt_ids = torch.arange(txt_start, txt_start + context.shape[1], device=x.device).reshape(1, -1, 1).repeat(x.shape[0], 1, 3)
    ids = torch.cat((txt_ids, img_ids), dim=1)
    image_rotary_emb = self.pe_embedder(ids).squeeze(1).unsqueeze(2).to(x.dtype)
    del ids, txt_ids, img_ids

    hidden_states = self.img_in(hidden_states)
    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    if guidance is not None:
        guidance = guidance * 1000

    temb = (
        self.time_text_embed(timestep, hidden_states)
        if guidance is None
        else self.time_text_embed(timestep, guidance, hidden_states)
    )

    patches_replace = transformer_options.get("patches_replace", {})
    patches = transformer_options.get("patches", {})
    cond_or_uncond = transformer_options.get("cond_or_uncond", [])
    blocks_replace = patches_replace.get("dit", {})
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
    if current_now['type'] == 'full':
        for i, block in enumerate(self.transformer_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["txt"], out["img"] = block(hidden_states=args["img"], encoder_hidden_states=args["txt"], encoder_hidden_states_mask=encoder_hidden_states_mask, temb=args["vec"], image_rotary_emb=args["pe"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": hidden_states, "txt": encoder_hidden_states, "vec": temb, "pe": image_rotary_emb}, {"original_block": block_wrap})
                hidden_states = out["img"]
                encoder_hidden_states = out["txt"]
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

            if "double_block" in patches:
                for p in patches["double_block"]:
                    out = p({"img": hidden_states, "txt": encoder_hidden_states, "x": x, "block_index": i})
                    hidden_states = out["img"]
                    encoder_hidden_states = out["txt"]

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        hidden_states[:, :add.shape[1]] += add

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        derivative_approximation(cache_dic=cache_dic_now, current=current_now, feature=hidden_states)
    elif current_now['type'] == 'Taylor':
        hidden_states = taylor_formula(cache_dic=cache_dic_now, current=current_now)
    hidden_states = hidden_states[:, :num_embeds].view(orig_shape[0], orig_shape[-2] // 2, orig_shape[-1] // 2, orig_shape[1], 2, 2)
    hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5)
    current_now["step"] += 1
    return hidden_states.reshape(orig_shape)[:, :, :, :x.shape[-2], :x.shape[-1]]

def cache_init_qwenimage(fresh_threshold, max_order, first_enhance, last_enhance, steps):   
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