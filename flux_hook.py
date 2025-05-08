from torch import Tensor
import torch
from comfy.ldm.flux.layers import timestep_embedding
from unittest.mock import patch
from typing import Dict
import math
from comfy.ldm.flux.math import attention
from contextlib import ExitStack
import comfy.model_management

def create_fluxpatches_dict(new_model):
    diffusion_model = new_model.get_model_object("diffusion_model")
    
    class ForwardPatcher:
        def __enter__(self):
            self.stack = ExitStack()
            
            # patch 主 forward 函数
            self.stack.enter_context(patch.object(
                diffusion_model,
                'forward_orig',
                taylorseer_flux_forward.__get__(diffusion_model, diffusion_model.__class__)
            ))
            
            # patch double_blocks 的 forward 函数
            for block in diffusion_model.double_blocks:
                self.stack.enter_context(patch.object(
                    block,
                    'forward',
                    flux_double_block_forward.__get__(block, block.__class__)
                ))
            
            # patch single_blocks 的 forward 函数
            for block in diffusion_model.single_blocks:
                self.stack.enter_context(patch.object(
                    block,
                    'forward',
                    flux_single_block_forward.__get__(block, block.__class__)
                ))
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.stack.close()
    
    return ForwardPatcher()

def taylorseer_flux_forward(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor = None,
    control = None,
    transformer_options={},
    attn_mask: Tensor = None,
) -> Tensor:
    patches_replace = transformer_options.get("patches_replace", {})
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    vec = vec + self.vector_in(y[:,:self.params.vec_in_dim])
    txt = self.txt_in(txt)

    if img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
    else:
        pe = None

    blocks_replace = patches_replace.get("dit", {})
    cal_type(cache_dic=self.cache_dic, current=self.current)
    for i, block in enumerate(self.double_blocks):
        self.current['layer'] = i
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(img=args["img"],
                                                        txt=args["txt"],
                                                        vec=args["vec"],
                                                        pe=args["pe"],
                                                        attn_mask=args.get("attn_mask"),
                                                        cache_dic=args.get("cache_dic"),
                                                        current=args.get("current"))
                return out

            out = blocks_replace[("double_block", i)]({"img": img,
                                                        "txt": txt,
                                                        "vec": vec,
                                                        "pe": pe,
                                                        "attn_mask": attn_mask,
                                                        "cache_dic": self.cache_dic,
                                                        "current": self.current},
                                                        {
                                                          "original_block": block_wrap,
                                                          "transformer_options": transformer_options
                                                        })
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = block(img=img,
                            txt=txt,
                            vec=vec,
                            pe=pe,
                            attn_mask=attn_mask,
                            cache_dic=self.cache_dic,
                            current=self.current)
        if control is not None: # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img += add

    img = torch.cat((txt, img), 1)

    for i, block in enumerate(self.single_blocks):
        self.current['layer'] = i
        if ("single_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(args["img"],
                                vec=args["vec"],
                                pe=args["pe"],
                                attn_mask=args.get("attn_mask"),
                                cache_dic=args.get("cache_dic"),
                                current=args.get("current"))
                return out

            out = blocks_replace[("single_block", i)]({"img": img,
                                                        "vec": vec,
                                                        "pe": pe,
                                                        "attn_mask": attn_mask,
                                                        "cache_dic": self.cache_dic,
                                                        "current": self.current},
                                                        {
                                                          "original_block": block_wrap,
                                                          "transformer_options": transformer_options
                                                        })
            img = out["img"]
        else:
            img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, cache_dic=self.cache_dic, current=self.current)

        if control is not None: # Controlnet
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1] :, ...] += add

    img = img[:, txt.shape[1] :, ...]

    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
    self.current["step"] += 1
    return img

def flux_double_block_forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, attn_mask=None, modulation_dims_img=None, modulation_dims_txt=None, cache_dic={}, current={}):

    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)

    current['stream'] = 'double_stream'
    if current['type'] == 'full':
        
        current['module'] = 'attn'
        taylor_cache_init(cache_dic=cache_dic, current=current)
        
        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        
        img_qkv = self.img_attn.qkv(img_modulated)
        
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)
        if self.flipped_img_txt:
            # run actual attention
            attn = attention(torch.cat((img_q, txt_q), dim=2),
                                torch.cat((img_k, txt_k), dim=2),
                                torch.cat((img_v, txt_v), dim=2),
                                pe=pe, mask=attn_mask)

            img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1]:]           
        else:
            # run actual attention
            attn = attention(torch.cat((txt_q, img_q), dim=2),
                            torch.cat((txt_k, img_k), dim=2),
                            torch.cat((txt_v, img_v), dim=2),
                            pe=pe, mask=attn_mask)
            
            txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]
        # calculate the img bloks
        current['module'] = 'img_attn'
        taylor_cache_init(cache_dic=cache_dic, current=current)
        img_attn = self.img_attn.proj(img_attn)
        
        derivative_approximation(cache_dic=cache_dic, current=current, feature=img_attn)
        
        img = img + img_mod1.gate * img_attn
        
        current['module'] = 'img_mlp'
        taylor_cache_init(cache_dic=cache_dic, current=current)
        img_mlp = self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
        derivative_approximation(cache_dic=cache_dic, current=current, feature=img_mlp)
        img = img + img_mod2.gate * img_mlp

        # calculate the txt bloks
        current['module'] = 'txt_attn'
        taylor_cache_init(cache_dic=cache_dic, current=current)
        txt_attn = self.txt_attn.proj(txt_attn)
        derivative_approximation(cache_dic=cache_dic, current=current, feature=txt_attn)
        txt += txt_mod1.gate * txt_attn

        current['module'] = 'txt_mlp'
        taylor_cache_init(cache_dic=cache_dic, current=current)
        txt_mlp = self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        derivative_approximation(cache_dic=cache_dic, current=current, feature=txt_mlp)
        txt += txt_mod2.gate * txt_mlp
        
    elif current['type'] == 'Taylor':
        
        current['module'] = 'attn'
        current['module'] = 'img_attn'
        img = img + img_mod1.gate * taylor_formula(cache_dic=cache_dic, current=current)
        
        current['module'] = 'img_mlp'
        img = img + img_mod2.gate * taylor_formula(cache_dic=cache_dic, current=current)
        
        current['module'] = 'txt_attn'
        txt += txt_mod1.gate * taylor_formula(cache_dic=cache_dic, current=current)
        
        current['module'] = 'txt_mlp'
        txt += txt_mod2.gate * taylor_formula(cache_dic=cache_dic, current=current)
        
    if txt.dtype == torch.float16:
        txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)
    return img, txt

def flux_single_block_forward(self, x: Tensor, vec: Tensor, pe: Tensor, attn_mask=None, cache_dic={}, current={}) -> Tensor:

    mod, _ = self.modulation(vec)

    current['stream'] = 'single_stream'
    if current['type'] == 'full':
        current['module'] = 'total'
        taylor_cache_init(cache_dic=cache_dic, current=current)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)
        # compute attention
        attn = attention(q, k, v, pe=pe, mask=attn_mask)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        derivative_approximation(cache_dic=cache_dic, current=current, feature=output)
    elif current['type'] == 'Taylor':
        current['module'] = 'total'
        output = taylor_formula(cache_dic=cache_dic, current=current)
    x += mod.gate * output
    if x.dtype == torch.float16:
        x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x

def cache_init_flux(fresh_threshold, max_order, first_enhance, last_enhance, steps):   
    '''
    Initialization for cache with memory optimization.
    '''
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1]={}
    cache_index[-1]={}
    cache_index['layer_index']={}
    cache_dic['attn_map'] = {}
    cache_dic['attn_map'][-1] = {}
    cache_dic['attn_map'][-1]['double_stream'] = {}
    cache_dic['attn_map'][-1]['single_stream'] = {}

    cache[-1]['double_stream']={}
    cache[-1]['single_stream']={}
    cache_dic['cache_counter'] = 0

    # Memory optimization: Only initialize structures when needed
    # Lazy initialization for double_stream
    for j in range(19):
        cache[-1]['double_stream'][j] = {}
        cache_index[-1][j] = {}
        # Defer initialization of detailed structures until needed
        cache_dic['attn_map'][-1]['double_stream'][j] = {}

    # Lazy initialization for single_stream
    for j in range(38):
        cache[-1]['single_stream'][j] = {}
        cache_index[-1][j] = {}
        # Defer initialization of detailed structures until needed
        cache_dic['attn_map'][-1]['single_stream'][j] = {}

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
    
    # Memory optimization settings
    cache_dic['memory_efficient'] = True
    cache_dic['importance_threshold'] = 0.01  # Threshold for selective caching
    cache_dic['max_cache_entries'] = 1000  # Limit total cache entries
    cache_dic['current_cache_size'] = 0  # Track current cache size
    cache_dic['prune_frequency'] = 5  # How often to prune the cache

    current = {}
    current['activated_steps'] = [0]
    current['step'] = 0
    current['num_steps'] = steps

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

    force_fresh = cache_dic['force_fresh']
    if not first_step:
        fresh_interval = cache_dic['cal_threshold']
    else:
        fresh_interval = cache_dic['fresh_threshold']

    if (first_step) or (cache_dic['cache_counter'] == fresh_interval - 1 ) or (current['step'] >= cache_dic['last_enhance'] - 1):
        current['type'] = 'full'
        cache_dic['cache_counter'] = 0
        current['activated_steps'].append(current['step'])
        #current['activated_times'].append(current['t'])
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

def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation with memory optimization.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    :param feature: The feature tensor to process
    """
    # Check if we should prune the cache based on frequency
    if cache_dic.get('memory_efficient', False) and current['step'] % cache_dic.get('prune_frequency', 5) == 0:
        prune_cache(cache_dic)
        
    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
    
    # Initialize with the current feature
    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature
    
    # Calculate higher-order derivatives
    for i in range(cache_dic['max_order']):
        if (cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
            # Calculate the next derivative
            next_derivative = (updated_taylor_factors[i] - cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i]) / difference_distance
            
            # Memory optimization: Skip storing derivatives with small magnitudes
            if cache_dic.get('memory_efficient', False):
                # Calculate importance based on magnitude
                importance = torch.mean(torch.abs(next_derivative)).item()
                
                # Only store if importance is above threshold
                if importance > cache_dic.get('importance_threshold', 0.01):
                    # Apply memory-efficient storage (half precision if possible)
                    if next_derivative.dtype == torch.float32 and not next_derivative.requires_grad:
                        next_derivative = next_derivative.half()
                    updated_taylor_factors[i + 1] = next_derivative
                else:
                    # Skip this derivative to save memory
                    break
            else:
                updated_taylor_factors[i + 1] = next_derivative
        else:
            break
    
    # Update the cache with new derivatives
    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors


def prune_cache(cache_dic: Dict):
    """
    Prune the cache to reduce memory usage by removing least important entries.
    
    :param cache_dic: Cache dictionary
    """
    if not cache_dic.get('memory_efficient', False):
        return
        
    # If we're under the limit, no need to prune
    if cache_dic['current_cache_size'] <= cache_dic['max_cache_entries']:
        return
        
    # Calculate how many entries to remove
    entries_to_remove = int(cache_dic['current_cache_size'] * 0.2)  # Remove 20% of entries
    
    # Collect all cache entries with their importance scores
    cache_entries = []
    
    # Traverse the cache structure
    for stream in cache_dic['cache'][-1]:
        for layer in cache_dic['cache'][-1][stream]:
            for module in cache_dic['cache'][-1][stream][layer]:
                entry_data = cache_dic['cache'][-1][stream][layer][module]
                
                # Calculate importance based on highest order derivative magnitude
                importance = 0
                for order in range(cache_dic['max_order'], -1, -1):
                    if order in entry_data:
                        importance = torch.mean(torch.abs(entry_data[order])).item()
                        break
                        
                cache_entries.append({
                    'stream': stream,
                    'layer': layer,
                    'module': module,
                    'importance': importance
                })
    
    # Sort by importance (ascending)
    cache_entries.sort(key=lambda x: x['importance'])
    
    # Remove least important entries
    for i in range(min(entries_to_remove, len(cache_entries))):
        entry = cache_entries[i]
        if entry['module'] in cache_dic['cache'][-1][entry['stream']][entry['layer']]:
            del cache_dic['cache'][-1][entry['stream']][entry['layer']][entry['module']]
            cache_dic['current_cache_size'] -= 1

def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Calculate the Taylor approximation with memory-efficient handling.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    :return: Approximated tensor
    """
    x = current['step'] - current['activated_steps'][-1]
    
    # Check if cache entry exists
    cache_entry = cache_dic['cache'][-1].get(current['stream'], {})
    if not cache_entry:
        # Return zero tensor with appropriate shape
        return torch.zeros(1)
        
    layer_entry = cache_entry.get(current['layer'], {})
    if not layer_entry:
        return torch.zeros(1)
        
    module_entry = layer_entry.get(current['module'], {})
    if not module_entry or 0 not in module_entry:
        return torch.zeros(1)
    
    # Initialize output with the zeroth order term
    output = module_entry[0].clone()
    
    # Calculate Taylor approximation using available derivatives
    for i in range(1, len(module_entry)):
        if i in module_entry:
            # Convert to original dtype if stored in half precision
            factor = module_entry[i]
            if factor.dtype != output.dtype:
                factor = factor.to(output.dtype)
                
            output += (1 / math.factorial(i)) * factor * (x ** i)
    
    return output

def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and allocate storage for different-order derivatives in the Taylor cache.
    Memory-optimized version that selectively initializes cache entries.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    # Only initialize cache if taylor_cache is enabled and we're at step 0
    if (current['step'] == 0) and (cache_dic['taylor_cache']):
        # Check if we're under the maximum cache entries limit
        if cache_dic.get('memory_efficient', False) and cache_dic['current_cache_size'] >= cache_dic['max_cache_entries']:
            # Skip initialization for less important layers if we're over the limit
            return
            
        # Initialize the cache entry
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}
        
        # Track cache size
        if cache_dic.get('memory_efficient', False):
            cache_dic['current_cache_size'] += 1

def force_init(cache_dic, current, tokens):
    '''
    Initialization for Force Activation step.
    '''
    cache_dic['cache_index'][-1][current['layer']][current['module']] = torch.zeros(tokens.shape[0], tokens.shape[1], dtype=torch.int, device=tokens.device)

    #if current['layer'] == 0:
    #    cache_dic['cache_index']['layer_index'][current['module']] = torch.zeros(tokens.shape[0], tokens.shape[1], dtype=torch.int, device=tokens.device)