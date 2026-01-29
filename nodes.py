from .flux_hook import create_fluxpatches_dict, cache_init_flux
from .hidream_hook import create_hidream_patches_dict, cache_init_hidream
from .wanvideo_hook import create_wanvideo_patches_dict, cache_init_wanvideo
from .qwenimage_hook import create_qwenimage_patches_dict, cache_init_qwenimage
from .lumina_hook import create_lumina_patches_dict, cache_init_lumina
import gc
import comfy.model_management

def get_module_memory_mb(module):
    memory = 0
    for param in module.parameters():
        if param.data is not None:
            memory += param.nelement() * param.element_size()
    return memory / (1024 * 1024)  # Convert to MB

def flux_block_swap(model, double_block_swap, single_block_swap):
    i = 0
    total_offload_memory = 0
    for block in model.model.diffusion_model.double_blocks:
        i += 1
        if i > double_block_swap:
            block.to(comfy.model_management.get_torch_device(), non_blocking=True)
        else:
            block.to(comfy.model_management.unet_offload_device(),non_blocking=True)
            total_offload_memory += get_module_memory_mb(block)
    i = 0
    for block in model.model.diffusion_model.single_blocks:
        i += 1
        if i > single_block_swap:
            block.to(comfy.model_management.get_torch_device(), non_blocking=True)
        else:
            block.to(comfy.model_management.unet_offload_device(),non_blocking=True)
            total_offload_memory += get_module_memory_mb(block)
    print(f"total_offload_memory: {total_offload_memory} MB")
    comfy.model_management.soft_empty_cache()
    gc.collect()

def hidream_block_swap(model, double_block_swap, single_block_swap):
    i = 0
    total_offload_memory = 0
    for block in model.model.diffusion_model.double_stream_blocks:
        i += 1
        if i > double_block_swap:
            block.to(comfy.model_management.get_torch_device(), non_blocking=True)
        else:
            block.to(comfy.model_management.unet_offload_device(),non_blocking=True)
            total_offload_memory += get_module_memory_mb(block)
    i = 0
    for block in model.model.diffusion_model.single_stream_blocks:
        i += 1
        if i > single_block_swap:
            block.to(comfy.model_management.get_torch_device(), non_blocking=True)
        else:
            block.to(comfy.model_management.unet_offload_device(),non_blocking=True)
            total_offload_memory += get_module_memory_mb(block)
    print(f"total_offload_memory: {total_offload_memory} MB")
    comfy.model_management.soft_empty_cache()
    gc.collect()

def wanvideo_block_swap(model, double_block_swap):
    i = 0
    total_offload_memory = 0
    for block in model.model.diffusion_model.blocks:
        i += 1
        if i > double_block_swap:
            block.to(comfy.model_management.get_torch_device(), non_blocking=True)
        else:
            block.to(comfy.model_management.unet_offload_device(),non_blocking=True)
            total_offload_memory += get_module_memory_mb(block)
    print(f"total_offload_memory: {total_offload_memory} MB")
    comfy.model_management.soft_empty_cache()
    gc.collect()

def qwenimage_block_swap(model, double_block_swap):
    i = 0
    total_offload_memory = 0
    for block in model.model.diffusion_model.transformer_blocks:
        i += 1
        if i > double_block_swap:
            block.to(comfy.model_management.get_torch_device(), non_blocking=True)
        else:
            block.to(comfy.model_management.unet_offload_device(),non_blocking=True)
            total_offload_memory += get_module_memory_mb(block)
    print(f"total_offload_memory: {total_offload_memory} MB")
    comfy.model_management.soft_empty_cache()
    gc.collect()

def cleanup_last_model(del_id):
    to_delete = []
    for i in range(len(comfy.model_management.current_loaded_models)):
        if id(comfy.model_management.current_loaded_models[i].model.model) == del_id:
            to_delete = [i] + to_delete
    for i in to_delete:
        x = comfy.model_management.current_loaded_models.pop(i)
        del x
    gc.collect()
    comfy.model_management.soft_empty_cache()

class FluxBlockSwap:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "double_block_swap": ("INT", {"default": 0, "min": 0, "max": 19, "step": 1, "tooltip": "Double block swap."}),
                "single_block_swap": ("INT", {"default": 0, "min": 0, "max": 38, "step": 1, "tooltip": "Single block swap."})
            }
        }
    
    RETURN_TYPES = ("BLOCKSWAPARGS",)
    RETURN_NAMES = ("block_swap_args",)
    FUNCTION = "setargs"
    CATEGORY = "TaylorSeer"
    TITLE = "FluxBlockSwap"

    def setargs(self, **kwargs):
        return (kwargs, )

class HidreamBlockSwap:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "double_block_swap": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1, "tooltip": "Double block swap."}),
                "single_block_swap": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1, "tooltip": "Single block swap."})
            }
        }
    
    RETURN_TYPES = ("BLOCKSWAPARGS",)
    RETURN_NAMES = ("block_swap_args",)
    FUNCTION = "setargs"
    CATEGORY = "TaylorSeer"
    TITLE = "HidreamBlockSwap"

    def setargs(self, **kwargs):
        return (kwargs, )

class WanvideoBlockSwap:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "double_block_swap": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1, "tooltip": "Double block swap."})
            }
        }
    
    RETURN_TYPES = ("BLOCKSWAPARGS",)
    RETURN_NAMES = ("block_swap_args",)
    FUNCTION = "setargs"
    CATEGORY = "TaylorSeer"
    TITLE = "WanvideoBlockSwap"

    def setargs(self, **kwargs):
        return (kwargs, )

class QwenimageBlockSwap:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "double_block_swap": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1, "tooltip": "Double block swap."})
            }
        }

    RETURN_TYPES = ("BLOCKSWAPARGS",)
    RETURN_NAMES = ("block_swap_args",)
    FUNCTION = "setargs"
    CATEGORY = "TaylorSeer"
    TITLE = "QwenimageBlockSwap"

    def setargs(self, **kwargs):
        return (kwargs, )

class TaylorSeer:

    last_model = {"id": None, "type": None}
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the TaylorSeer will be applied to."}),
                "model_type": (["flux", "hidream"], {"default": "flux", "tooltip": "Supported diffusion model."}),
                "fresh_threshold": ("INT", {"default": 6, "min": 3, "max": 7, "step": 1, "tooltip": "Fresh threshold."}),
                "max_order": ("INT", {"default": 2, "min": 0, "max": 2, "step": 1, "tooltip": "Max order."}),
                "first_enhance": ("INT", {"default": 3, "min": 0, "max": 100, "step": 1, "tooltip": "First enhance."}),
                "last_enhance": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1, "tooltip": "Last enhance."})
            },
            "optional": {
                "block_swap_args": ("BLOCKSWAPARGS", ),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_taylorseer"
    CATEGORY = "TaylorSeer"
    TITLE = "TaylorSeer"
    
    def apply_taylorseer(self, model, model_type: str, fresh_threshold: int, max_order: int, first_enhance: int, last_enhance: int, block_swap_args: dict = {}):
        new_model = model.clone()
        if 'transformer_options' not in new_model.model_options:
            new_model.model_options['transformer_options'] = {} 
        
        if "flux" in model_type:
            context = create_fluxpatches_dict(new_model)
            # Check if need to cleanup last model
            if self.last_model["type"] == "flux" and self.last_model["id"] != id(new_model.model):
                cleanup_last_model(self.last_model["id"])
            self.last_model = {"id": id(new_model.model), "type": "flux"}
        elif "hidream" in model_type:
            context = create_hidream_patches_dict(new_model)
            # Check if need to cleanup last model
            if self.last_model["type"] == "hidream" and self.last_model["id"] != id(new_model.model):
                cleanup_last_model(self.last_model["id"])
            self.last_model = {"id": id(new_model.model), "type": "hidream"}
        else:
            raise ValueError(f"Unknown type {model_type}")
        
        def unet_wrapper_function(model_function, kwargs):
            input = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]
            # check current step
            sigmas = c["transformer_options"]["sample_sigmas"]
            if "flux" in model_type:
                double_block_swap = block_swap_args.get("double_block_swap", 0)
                single_block_swap = block_swap_args.get("single_block_swap", 0)
            elif "hidream" in model_type:
                double_block_swap = block_swap_args.get("double_block_swap", 0)
                single_block_swap = block_swap_args.get("single_block_swap", 0)
            matched_step_index = (sigmas == timestep[0]).nonzero()
            if len(matched_step_index) > 0:
                current_step_index = matched_step_index.item()
            else:
                current_step_index = 0
                for i in range(len(sigmas) - 1):
                    if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                        current_step_index = i
                        break
            # init cache
            if current_step_index == 0:
                if "flux" in model_type:
                    new_model.model.diffusion_model.cache_dic, new_model.model.diffusion_model.current = cache_init_flux(fresh_threshold, max_order, first_enhance, last_enhance, len(sigmas))
                    flux_block_swap(new_model, double_block_swap, single_block_swap)
                elif "hidream" in model_type:
                    new_model.model.diffusion_model.cache_dic, new_model.model.diffusion_model.current = cache_init_hidream(fresh_threshold, max_order, first_enhance, last_enhance, len(sigmas))
                    hidream_block_swap(new_model, double_block_swap, single_block_swap)
            with context:
                result = model_function(input, timestep, **c)
            if current_step_index == len(sigmas) - 2:
                del new_model.model.diffusion_model.cache_dic
                del new_model.model.diffusion_model.current
                gc.collect()
                comfy.model_management.soft_empty_cache()
                if "flux" in model_type:
                    if double_block_swap + single_block_swap > 0:
                        for block in new_model.model.diffusion_model.double_blocks:
                            block.to(comfy.model_management.get_torch_device(), non_blocking=True)
                        for block in new_model.model.diffusion_model.single_blocks:
                            block.to(comfy.model_management.get_torch_device(), non_blocking=True)
                elif "hidream" in model_type:
                    if double_block_swap + single_block_swap > 0:
                        for block in new_model.model.diffusion_model.double_stream_blocks:
                            block.to(comfy.model_management.get_torch_device(), non_blocking=True)
                        for block in new_model.model.diffusion_model.single_stream_blocks:
                            block.to(comfy.model_management.get_torch_device(), non_blocking=True)
            return result

        new_model.set_model_unet_function_wrapper(unet_wrapper_function)

        return (new_model,)

class TaylorSeerLite:

    last_model = {"id": None, "type": None}
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the TaylorSeer will be applied to."}),
                "model_type": (["flux", "flux2", "hidream", "wanvideo", "qwenimage", "lumina2"], {"default": "flux", "tooltip": "Supported diffusion model."}),
                "fresh_threshold": ("INT", {"default": 6, "min": 3, "max": 7, "step": 1, "tooltip": "Fresh threshold."}),
                "max_order": ("INT", {"default": 2, "min": 0, "max": 2, "step": 1, "tooltip": "Max order."}),
                "first_enhance": ("INT", {"default": 3, "min": 0, "max": 100, "step": 1, "tooltip": "First enhance."}),
                "last_enhance": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1, "tooltip": "Last enhance."})
            },
            "optional": {
                "block_swap_args": ("BLOCKSWAPARGS", ),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_taylorseer_lite"
    CATEGORY = "TaylorSeer"
    TITLE = "TaylorSeer"
    
    def apply_taylorseer_lite(self, model, model_type: str, fresh_threshold: int, max_order: int, first_enhance: int, last_enhance: int, block_swap_args: dict = {}):
        new_model = model.clone()
        if 'transformer_options' not in new_model.model_options:
            new_model.model_options['transformer_options'] = {} 
        
        if "flux" in model_type:
            context = create_fluxpatches_dict(new_model, type="lite")
            # Check if need to cleanup last model
            if self.last_model["type"] == "flux" and self.last_model["id"] != id(new_model.model):
                cleanup_last_model(self.last_model["id"])
            self.last_model = {"id": id(new_model.model), "type": "flux"}
        elif "hidream" in model_type:
            context = create_hidream_patches_dict(new_model, type="lite")
            # Check if need to cleanup last model
            if self.last_model["type"] == "hidream" and self.last_model["id"] != id(new_model.model):
                cleanup_last_model(self.last_model["id"])
            self.last_model = {"id": id(new_model.model), "type": "hidream"}
        elif "wanvideo" in model_type:
            context = create_wanvideo_patches_dict(new_model, type="lite")
            # Check if need to cleanup last model
            if self.last_model["type"] == "wanvideo" and self.last_model["id"] != id(new_model.model):
                cleanup_last_model(self.last_model["id"])
            self.last_model = {"id": id(new_model.model), "type": "wanvideo"}
        elif "qwenimage" in model_type:
            context = create_qwenimage_patches_dict(new_model, type="lite")
            # Check if need to cleanup last model
            if self.last_model["type"] == "qwenimage" and self.last_model["id"] != id(new_model.model):
                cleanup_last_model(self.last_model["id"])
            self.last_model = {"id": id(new_model.model), "type": "qwenimage"}
        elif "lumina2" in model_type:
            context = create_lumina_patches_dict(new_model, type="lite")
            # Check if need to cleanup last model
            if self.last_model["type"] == "lumina2" and self.last_model["id"] != id(new_model.model):
                cleanup_last_model(self.last_model["id"])
            self.last_model = {"id": id(new_model.model), "type": "lumina2"}
        else:
            raise ValueError(f"Unknown type {model_type}")
        
        def unet_wrapper_function(model_function, kwargs):
            input = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]
            # check current step
            sigmas = c["transformer_options"]["sample_sigmas"]
            cond_or_uncond = c["transformer_options"].get("cond_or_uncond", [])
            if "flux" in model_type:
                double_block_swap = block_swap_args.get("double_block_swap", 0)
                single_block_swap = block_swap_args.get("single_block_swap", 0)
            elif "hidream" in model_type:
                double_block_swap = block_swap_args.get("double_block_swap", 0)
                single_block_swap = block_swap_args.get("single_block_swap", 0)
            elif "wanvideo" in model_type:
                double_block_swap = block_swap_args.get("double_block_swap", 0)
            elif "qwenimage" in model_type:
                double_block_swap = block_swap_args.get("double_block_swap", 0)
            elif "lumina2" in model_type:
                double_block_swap = block_swap_args.get("double_block_swap", 0)
            matched_step_index = (sigmas == timestep[0]).nonzero()
            if len(matched_step_index) > 0:
                current_step_index = matched_step_index.item()
            else:
                current_step_index = 0
                for i in range(len(sigmas) - 1):
                    if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                        current_step_index = i
                        break
            # init cache
            if current_step_index == 0 or hasattr(new_model.model.diffusion_model, 'cache_dic') is False:
                if "flux2" in model_type:
                    new_model.model.diffusion_model.cache_dic, new_model.model.diffusion_model.current = cache_init_flux(fresh_threshold, max_order, first_enhance, last_enhance, len(sigmas))
                elif "flux" in model_type:
                    new_model.model.diffusion_model.cache_dic, new_model.model.diffusion_model.current = cache_init_flux(fresh_threshold, max_order, first_enhance, last_enhance, len(sigmas))
                    flux_block_swap(new_model, double_block_swap, single_block_swap)
                elif "hidream" in model_type:
                    new_model.model.diffusion_model.cache_dic, new_model.model.diffusion_model.current = cache_init_hidream(fresh_threshold, max_order, first_enhance, last_enhance, len(sigmas))
                    hidream_block_swap(new_model, double_block_swap, single_block_swap)
                elif "wanvideo" in model_type:
                    new_model.model.diffusion_model.cache_dic, new_model.model.diffusion_model.current = cache_init_wanvideo(fresh_threshold, max_order, first_enhance, last_enhance, len(sigmas))
                    wanvideo_block_swap(new_model, double_block_swap)
                elif "qwenimage" in model_type and current_step_index != len(sigmas) - 2:
                    if cond_or_uncond == [1]:
                        new_model.model.diffusion_model.cache_dic_negative, new_model.model.diffusion_model.current_negative = cache_init_qwenimage(fresh_threshold, max_order, first_enhance, last_enhance, len(sigmas))
                    else:
                        new_model.model.diffusion_model.cache_dic, new_model.model.diffusion_model.current = cache_init_qwenimage(fresh_threshold, max_order, first_enhance, last_enhance, len(sigmas))
                    qwenimage_block_swap(new_model, double_block_swap)
                elif "lumina2" in model_type and current_step_index != len(sigmas) - 2:
                    if cond_or_uncond == [1]:
                        new_model.model.diffusion_model.cache_dic_negative, new_model.model.diffusion_model.current_negative = cache_init_lumina(fresh_threshold, max_order, first_enhance, last_enhance, len(sigmas))
                    else:
                        new_model.model.diffusion_model.cache_dic, new_model.model.diffusion_model.current = cache_init_lumina(fresh_threshold, max_order, first_enhance, last_enhance, len(sigmas))
            with context:
                result = model_function(input, timestep, **c)
            if current_step_index == len(sigmas) - 2:
                if model_type in ["qwenimage", "lumina2"] and cond_or_uncond == [1]:
                    del new_model.model.diffusion_model.cache_dic_negative
                    del new_model.model.diffusion_model.current_negative
                else:
                    del new_model.model.diffusion_model.cache_dic
                    del new_model.model.diffusion_model.current
                gc.collect()
                comfy.model_management.soft_empty_cache()
                if "flux" in model_type:
                    if double_block_swap + single_block_swap > 0:
                        for block in new_model.model.diffusion_model.double_blocks:
                            block.to(comfy.model_management.get_torch_device(), non_blocking=True)
                        for block in new_model.model.diffusion_model.single_blocks:
                            block.to(comfy.model_management.get_torch_device(), non_blocking=True)
                elif "hidream" in model_type:
                    if double_block_swap + single_block_swap > 0:
                        for block in new_model.model.diffusion_model.double_stream_blocks:
                            block.to(comfy.model_management.get_torch_device(), non_blocking=True)
                        for block in new_model.model.diffusion_model.single_stream_blocks:
                            block.to(comfy.model_management.get_torch_device(), non_blocking=True)
                elif "wanvideo" in model_type:
                    if double_block_swap > 0:
                        for block in new_model.model.diffusion_model.blocks:
                            block.to(comfy.model_management.get_torch_device(), non_blocking=True)
                elif "qwenimage" in model_type:
                    if double_block_swap > 0:
                        for block in new_model.model.diffusion_model.transformer_blocks:
                            block.to(comfy.model_management.get_torch_device(), non_blocking=True)
            return result

        new_model.set_model_unet_function_wrapper(unet_wrapper_function)

        return (new_model,)

NODE_CLASS_MAPPINGS = {
    "TaylorSeer": TaylorSeer,
    "FluxBlockSwap": FluxBlockSwap,
    "HidreamBlockSwap": HidreamBlockSwap,
    "WanvideoBlockSwap": WanvideoBlockSwap,
    "QwenimageBlockSwap": QwenimageBlockSwap,
    "TaylorSeerLite": TaylorSeerLite
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TaylorSeer": "TaylorSeer",
    "FluxBlockSwap": "FluxBlockSwap",
    "HidreamBlockSwap": "HidreamBlockSwap",
    "WanvideoBlockSwap": "WanvideoBlockSwap",
    "QwenimageBlockSwap": "QwenimageBlockSwap",
    "TaylorSeerLite": "TaylorSeerLite"
}