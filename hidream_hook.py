from torch import Tensor
import torch
from contextlib import ExitStack
from unittest.mock import patch
from typing import Optional
import comfy.ldm.common_dit
from einops import repeat
from typing import Dict
import math

def create_hidream_patches_dict(new_model, type="full"):
    diffusion_model = new_model.get_model_object("diffusion_model")
    
    class ForwardPatcher:
        def __enter__(self):
            self.stack = ExitStack()
            
            if type == "full":
                # patch 主 forward 函数
                self.stack.enter_context(patch.object(
                    diffusion_model,
                    'forward',
                    taylorseer_hidream_forward.__get__(diffusion_model, diffusion_model.__class__)
                ))

                # patch double_blocks 的 forward 函数
                for block in diffusion_model.double_stream_blocks:
                    self.stack.enter_context(patch.object(
                        block,
                        'forward',
                        taylorseer_hidream_block_forward.__get__(block, block.__class__)
                    ))
                    self.stack.enter_context(patch.object(
                        block.block,
                        'forward',
                        hidream_image_transformer_block_forward.__get__(block.block, block.block.__class__)
                    ))
                
                # patch single_blocks 的 forward 函数
                for block in diffusion_model.single_stream_blocks:
                    self.stack.enter_context(patch.object(
                        block,
                        'forward',
                        taylorseer_hidream_block_forward.__get__(block, block.__class__)
                    ))
                    self.stack.enter_context(patch.object(
                        block.block,
                        'forward',
                        hidream_image_single_transformer_block_forward.__get__(block.block, block.block.__class__)
                    ))
            elif type == "lite":
                # patch Taylor 的 forward 函数
                self.stack.enter_context(patch.object(
                    diffusion_model,
                    'forward_orig',
                    taylorseer_lite_hidream_forward.__get__(diffusion_model, diffusion_model.__class__)
                ))
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.stack.close()
    
    return ForwardPatcher()

def taylorseer_hidream_block_forward(
    self,
    image_tokens: torch.FloatTensor,
    image_tokens_masks: Optional[torch.FloatTensor] = None,
    text_tokens: Optional[torch.FloatTensor] = None,
    adaln_input: torch.FloatTensor = None,
    rope: torch.FloatTensor = None,
    cache_dic: Dict = None, ##
    current: Dict = None, ##
) -> torch.FloatTensor:
    return self.block(
        image_tokens,
        image_tokens_masks,
        text_tokens,
        adaln_input,
        rope,
        cache_dic,
        current,
    )

def taylorseer_hidream_forward(
    self,
    x: torch.Tensor,
    t: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    context: Optional[torch.Tensor] = None,
    encoder_hidden_states_llama3=None,
    image_cond=None,
    control = None,
    transformer_options = {},
) -> torch.Tensor:
    bs, c, h, w = x.shape
    if image_cond is not None:
        x = torch.cat([x, image_cond], dim=-1)
    hidden_states = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
    timesteps = t
    pooled_embeds = y
    T5_encoder_hidden_states = context

    img_sizes = None

    # spatial forward
    batch_size = hidden_states.shape[0]
    hidden_states_type = hidden_states.dtype

    # 0. time
    timesteps = self.expand_timesteps(timesteps, batch_size, hidden_states.device)
    timesteps = self.t_embedder(timesteps, hidden_states_type)
    p_embedder = self.p_embedder(pooled_embeds)
    adaln_input = timesteps + p_embedder

    hidden_states, image_tokens_masks, img_sizes = self.patchify(hidden_states, self.max_seq, img_sizes)
    if image_tokens_masks is None:
        pH, pW = img_sizes[0]
        img_ids = torch.zeros(pH, pW, 3, device=hidden_states.device)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH, device=hidden_states.device)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW, device=hidden_states.device)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
    hidden_states = self.x_embedder(hidden_states)

    # T5_encoder_hidden_states = encoder_hidden_states[0]
    encoder_hidden_states = encoder_hidden_states_llama3.movedim(1, 0)
    encoder_hidden_states = [encoder_hidden_states[k] for k in self.llama_layers]

    if self.caption_projection is not None:
        new_encoder_hidden_states = []
        for i, enc_hidden_state in enumerate(encoder_hidden_states):
            enc_hidden_state = self.caption_projection[i](enc_hidden_state)
            enc_hidden_state = enc_hidden_state.view(batch_size, -1, hidden_states.shape[-1])
            new_encoder_hidden_states.append(enc_hidden_state)
        encoder_hidden_states = new_encoder_hidden_states
        T5_encoder_hidden_states = self.caption_projection[-1](T5_encoder_hidden_states)
        T5_encoder_hidden_states = T5_encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
        encoder_hidden_states.append(T5_encoder_hidden_states)

    txt_ids = torch.zeros(
        batch_size,
        encoder_hidden_states[-1].shape[1] + encoder_hidden_states[-2].shape[1] + encoder_hidden_states[0].shape[1],
        3,
        device=img_ids.device, dtype=img_ids.dtype
    )
    ids = torch.cat((img_ids, txt_ids), dim=1)
    rope = self.pe_embedder(ids)

    cal_type(cache_dic=self.cache_dic, current=self.current)
    # 2. Blocks
    block_id = 0
    self.current['layer'] = 0
    initial_encoder_hidden_states = torch.cat([encoder_hidden_states[-1], encoder_hidden_states[-2]], dim=1)
    initial_encoder_hidden_states_seq_len = initial_encoder_hidden_states.shape[1]
    for bid, block in enumerate(self.double_stream_blocks):
        self.current['stream'] = 'double_stream'
        self.current['layer'] = bid
        cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
        cur_encoder_hidden_states = torch.cat([initial_encoder_hidden_states, cur_llama31_encoder_hidden_states], dim=1)
        hidden_states, initial_encoder_hidden_states = block(
            image_tokens = hidden_states,
            image_tokens_masks = image_tokens_masks,
            text_tokens = cur_encoder_hidden_states,
            adaln_input = adaln_input,
            rope = rope,
            cache_dic = self.cache_dic,
            current = self.current
        )
        initial_encoder_hidden_states = initial_encoder_hidden_states[:, :initial_encoder_hidden_states_seq_len]
        block_id += 1

    image_tokens_seq_len = hidden_states.shape[1]
    hidden_states = torch.cat([hidden_states, initial_encoder_hidden_states], dim=1)
    hidden_states_seq_len = hidden_states.shape[1]
    if image_tokens_masks is not None:
        encoder_attention_mask_ones = torch.ones(
            (batch_size, initial_encoder_hidden_states.shape[1] + cur_llama31_encoder_hidden_states.shape[1]),
            device=image_tokens_masks.device, dtype=image_tokens_masks.dtype
        )
        image_tokens_masks = torch.cat([image_tokens_masks, encoder_attention_mask_ones], dim=1)

    for bid, block in enumerate(self.single_stream_blocks):
        self.current['stream'] = 'single_stream'
        self.current['layer'] = bid
        cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
        hidden_states = torch.cat([hidden_states, cur_llama31_encoder_hidden_states], dim=1)
        hidden_states = block(
            image_tokens=hidden_states,
            image_tokens_masks=image_tokens_masks,
            text_tokens=None,
            adaln_input=adaln_input,
            rope=rope,
            cache_dic=self.cache_dic,
            current=self.current
        )
        hidden_states = hidden_states[:, :hidden_states_seq_len]
        block_id += 1

    hidden_states = hidden_states[:, :image_tokens_seq_len, ...]
    output = self.final_layer(hidden_states, adaln_input)
    output = self.unpatchify(output, img_sizes)
    self.current["step"] += 1
    return -output[:, :, :h, :w]

def taylorseer_lite_hidream_forward(
    self,
    x: torch.Tensor,
    t: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    context: Optional[torch.Tensor] = None,
    encoder_hidden_states_llama3=None,
    image_cond=None,
    control = None,
    transformer_options = {},
) -> torch.Tensor:
    bs, c, h, w = x.shape
    if image_cond is not None:
        x = torch.cat([x, image_cond], dim=-1)
    hidden_states = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
    timesteps = t
    pooled_embeds = y
    T5_encoder_hidden_states = context

    img_sizes = None

    # spatial forward
    batch_size = hidden_states.shape[0]
    hidden_states_type = hidden_states.dtype

    # 0. time
    timesteps = self.expand_timesteps(timesteps, batch_size, hidden_states.device)
    timesteps = self.t_embedder(timesteps, hidden_states_type)
    p_embedder = self.p_embedder(pooled_embeds)
    adaln_input = timesteps + p_embedder

    hidden_states, image_tokens_masks, img_sizes = self.patchify(hidden_states, self.max_seq, img_sizes)
    if image_tokens_masks is None:
        pH, pW = img_sizes[0]
        img_ids = torch.zeros(pH, pW, 3, device=hidden_states.device)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH, device=hidden_states.device)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW, device=hidden_states.device)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
    hidden_states = self.x_embedder(hidden_states)

    # T5_encoder_hidden_states = encoder_hidden_states[0]
    encoder_hidden_states = encoder_hidden_states_llama3.movedim(1, 0)
    encoder_hidden_states = [encoder_hidden_states[k] for k in self.llama_layers]

    if self.caption_projection is not None:
        new_encoder_hidden_states = []
        for i, enc_hidden_state in enumerate(encoder_hidden_states):
            enc_hidden_state = self.caption_projection[i](enc_hidden_state)
            enc_hidden_state = enc_hidden_state.view(batch_size, -1, hidden_states.shape[-1])
            new_encoder_hidden_states.append(enc_hidden_state)
        encoder_hidden_states = new_encoder_hidden_states
        T5_encoder_hidden_states = self.caption_projection[-1](T5_encoder_hidden_states)
        T5_encoder_hidden_states = T5_encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
        encoder_hidden_states.append(T5_encoder_hidden_states)

    txt_ids = torch.zeros(
        batch_size,
        encoder_hidden_states[-1].shape[1] + encoder_hidden_states[-2].shape[1] + encoder_hidden_states[0].shape[1],
        3,
        device=img_ids.device, dtype=img_ids.dtype
    )
    ids = torch.cat((img_ids, txt_ids), dim=1)
    rope = self.pe_embedder(ids)

    cal_type(cache_dic=self.cache_dic, current=self.current)
    # 2. Blocks
    block_id = 0
    self.current['stream'] = 'final_stream'
    self.current['layer'] = 0
    self.current['module'] = 'final'
    if self.current['type'] == 'full':
        initial_encoder_hidden_states = torch.cat([encoder_hidden_states[-1], encoder_hidden_states[-2]], dim=1)
        initial_encoder_hidden_states_seq_len = initial_encoder_hidden_states.shape[1]
        for bid, block in enumerate(self.double_stream_blocks):
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
            cur_encoder_hidden_states = torch.cat([initial_encoder_hidden_states, cur_llama31_encoder_hidden_states], dim=1)
            hidden_states, initial_encoder_hidden_states = block(
                image_tokens = hidden_states,
                image_tokens_masks = image_tokens_masks,
                text_tokens = cur_encoder_hidden_states,
                adaln_input = adaln_input,
                rope = rope,
            )
            initial_encoder_hidden_states = initial_encoder_hidden_states[:, :initial_encoder_hidden_states_seq_len]
            block_id += 1

        image_tokens_seq_len = hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, initial_encoder_hidden_states], dim=1)
        hidden_states_seq_len = hidden_states.shape[1]
        if image_tokens_masks is not None:
            encoder_attention_mask_ones = torch.ones(
                (batch_size, initial_encoder_hidden_states.shape[1] + cur_llama31_encoder_hidden_states.shape[1]),
                device=image_tokens_masks.device, dtype=image_tokens_masks.dtype
            )
            image_tokens_masks = torch.cat([image_tokens_masks, encoder_attention_mask_ones], dim=1)

        for bid, block in enumerate(self.single_stream_blocks):
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
            hidden_states = torch.cat([hidden_states, cur_llama31_encoder_hidden_states], dim=1)
            hidden_states = block(
                image_tokens=hidden_states,
                image_tokens_masks=image_tokens_masks,
                text_tokens=None,
                adaln_input=adaln_input,
                rope=rope,
            )
            hidden_states = hidden_states[:, :hidden_states_seq_len]
            block_id += 1

        hidden_states = hidden_states[:, :image_tokens_seq_len, ...]
        output = self.final_layer(hidden_states, adaln_input)
        derivative_approximation(cache_dic=self.cache_dic, current=self.current, feature=output)
    elif self.current['type'] == 'taylor_cache':
        output = taylor_formula(cache_dic=self.cache_dic, current=self.current)
    output = self.unpatchify(output, img_sizes)
    self.current["step"] += 1
    return -output[:, :, :h, :w]

def hidream_image_transformer_block_forward(
    self,
    image_tokens: torch.FloatTensor,
    image_tokens_masks: Optional[torch.FloatTensor] = None,
    text_tokens: Optional[torch.FloatTensor] = None,
    adaln_input: Optional[torch.FloatTensor] = None,
    rope: torch.FloatTensor = None,
    cache_dic = {}, 
    current = {}
) -> torch.FloatTensor:
    if current['type'] == 'full':
        wtype = image_tokens.dtype
        shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i, \
        shift_msa_t, scale_msa_t, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = \
            self.adaLN_modulation(adaln_input)[:,None].chunk(12, dim=-1)

        # 1. MM-Attention
        norm_image_tokens = self.norm1_i(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_msa_i) + shift_msa_i
        norm_text_tokens = self.norm1_t(text_tokens).to(dtype=wtype)
        norm_text_tokens = norm_text_tokens * (1 + scale_msa_t) + shift_msa_t

        attn_output_i, attn_output_t = self.attn1(
            norm_image_tokens,
            image_tokens_masks,
            norm_text_tokens,
            rope = rope,
        )
        current['module'] = 'img_attn'
        taylor_cache_init(cache_dic, current)
        derivative_approximation(cache_dic, current, attn_output_i)
        
        current['module'] = 'txt_attn'
        taylor_cache_init(cache_dic, current)
        derivative_approximation(cache_dic, current, attn_output_t)

        image_tokens = gate_msa_i * attn_output_i + image_tokens
        text_tokens = gate_msa_t * attn_output_t + text_tokens

        # 2. Feed-forward
        norm_image_tokens = self.norm3_i(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_mlp_i) + shift_mlp_i
        norm_text_tokens = self.norm3_t(text_tokens).to(dtype=wtype)
        norm_text_tokens = norm_text_tokens * (1 + scale_mlp_t) + shift_mlp_t

        mlp_output_img = self.ff_i(norm_image_tokens)
        mlp_output_txt = self.ff_t(norm_text_tokens)
        current['module'] = 'img_mlp'
        taylor_cache_init(cache_dic, current)
        derivative_approximation(cache_dic, current, mlp_output_img)

        current['module'] = 'txt_mlp'
        taylor_cache_init(cache_dic, current)
        derivative_approximation(cache_dic, current, mlp_output_txt)
        image_tokens = gate_mlp_i * mlp_output_img + image_tokens
        text_tokens = gate_mlp_t * mlp_output_txt + text_tokens
    elif current['type'] == 'taylor_cache':
        wtype = image_tokens.dtype
        shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i, \
        shift_msa_t, scale_msa_t, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = \
            self.adaLN_modulation(adaln_input)[:,None].chunk(12, dim=-1)
        current['module'] = 'img_attn'
        attn_output_i = taylor_formula(cache_dic=cache_dic, current=current)
        image_tokens = gate_msa_i * attn_output_i + image_tokens

        current['module'] = 'txt_attn'
        attn_output_t = taylor_formula(cache_dic=cache_dic, current=current)
        text_tokens = gate_msa_t * attn_output_t + text_tokens

        current['module'] = 'img_mlp'
        mlp_output_img = taylor_formula(cache_dic=cache_dic, current=current)
        image_tokens = gate_mlp_i * mlp_output_img + image_tokens

        current['module'] = 'txt_mlp'
        mlp_output_txt = taylor_formula(cache_dic=cache_dic, current=current)
        text_tokens = gate_mlp_t * mlp_output_txt + text_tokens
    return image_tokens, text_tokens

def hidream_image_single_transformer_block_forward(
    self,
    image_tokens: torch.FloatTensor,
    image_tokens_masks: Optional[torch.FloatTensor] = None,
    text_tokens: Optional[torch.FloatTensor] = None,
    adaln_input: Optional[torch.FloatTensor] = None,
    rope: torch.FloatTensor = None,
    cache_dic = {},
    current = {}
) -> torch.FloatTensor:
    if current["type"] == "full":
        wtype = image_tokens.dtype
        shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i = \
            self.adaLN_modulation(adaln_input)[:,None].chunk(6, dim=-1)

        # 1. MM-Attention
        norm_image_tokens = self.norm1_i(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_msa_i) + shift_msa_i
        attn_output_i = self.attn1(
            norm_image_tokens,
            image_tokens_masks,
            rope = rope,
        )
        current["module"] = "img_attn"
        taylor_cache_init(cache_dic=cache_dic, current=current)
        derivative_approximation(cache_dic=cache_dic, current=current, feature=attn_output_i)
        image_tokens = gate_msa_i * attn_output_i + image_tokens

        # 2. Feed-forward
        norm_image_tokens = self.norm3_i(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_mlp_i) + shift_mlp_i
        mlp_output = self.ff_i(norm_image_tokens.to(dtype=wtype))
        current["module"] = "img_mlp"
        taylor_cache_init(cache_dic=cache_dic, current=current)
        derivative_approximation(cache_dic=cache_dic, current=current, feature=mlp_output)
        image_tokens = gate_mlp_i * mlp_output + image_tokens
    elif current["type"] == "taylor_cache":
        shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i = \
                self.adaLN_modulation(adaln_input)[:,None].chunk(6, dim=-1)
        current["module"] = "img_attn"
        attn_output_i = taylor_formula(cache_dic=cache_dic, current=current)
        image_tokens = gate_msa_i * attn_output_i + image_tokens
        current["module"] = "img_mlp"
        mlp_output = taylor_formula(cache_dic=cache_dic, current=current)
        image_tokens = gate_mlp_i * mlp_output + image_tokens
    return image_tokens

def cache_init_hidream(fresh_threshold, max_order, first_enhance, last_enhance, steps):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1]={}
    cache_index[-1]={}
    cache_index['layer_index']={}

    cache[-1]['double_stream']={}
    cache[-1]['single_stream']={}
    cache[-1]['final_stream'] = {}
    cache_dic['cache_counter'] = 0

    final_stream_modules = ['final']

    for j in range(16):
        cache[-1]['double_stream'][j] = {}
        cache_index[-1][j] = {}

    for j in range(32):
        cache[-1]['single_stream'][j] = {}
        cache_index[-1][j] = {}
    
    cache[-1]['final_stream'][0] = {}
    for module in final_stream_modules:
        cache[-1]['final_stream'][0][module] = {}
    cache_index[-1][0] = {}

    cache_dic['cache_index'] = cache_index
    cache_dic['cache'] = cache
    cache_dic['fresh_threshold'] = fresh_threshold
    cache_dic['max_order'] = max_order
    cache_dic['first_enhance'] = first_enhance
    cache_dic['last_enhance'] = last_enhance

    current = {}
    current['activated_steps'] = [0]
    current['step'] = 0
    current['num_steps'] = steps

    return cache_dic, current

def cal_type(cache_dic, current):
    '''
    Determine calculation type for this step
    '''

    first_step = (current['step'] < cache_dic['first_enhance'])

    fresh_interval = cache_dic['fresh_threshold']

    if (first_step) or (cache_dic['cache_counter'] == fresh_interval - 1 ) or (current['step'] >= cache_dic['last_enhance'] - 1):
        current['type'] = 'full'
        cache_dic['cache_counter'] = 0
        current['activated_steps'].append(current['step'])
    else:
        cache_dic['cache_counter'] += 1
        current['type'] = 'taylor_cache'

def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic['max_order']):
        if (cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i]) / difference_distance
        else:
            break
    
    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors


def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    x = current['step'] - current['activated_steps'][-1]
    output = 0
    
    for i in range(len(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']])):
        output += (1 / math.factorial(i)) * cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i] * (x ** i)
    
    return output

def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache, expanding storage areas for Taylor series derivatives
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if current['step'] == 0:
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}