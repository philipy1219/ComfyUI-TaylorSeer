from .flux_hook import cal_type,create_fluxpatches_dict,cache_init_flux

class TaylorSeer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the TaylorSeer will be applied to."}),
                "model_type": (["flux"], {"default": "flux", "tooltip": "Supported diffusion model."}),
                "fresh_threshold": ("INT", {"default": 6, "min": 3, "max": 7, "step": 1, "tooltip": "Fresh threshold."}),
                "max_order": ("INT", {"default": 2, "min": 0, "max": 2, "step": 1, "tooltip": "Max order."}),
                "first_enhance": ("INT", {"default": 3, "min": 0, "max": 100, "step": 1, "tooltip": "First enhance."}),
                "last_enhance": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1, "tooltip": "Last enhance."})
            },
            "optional": {
                "memory_efficient": (["enabled", "disabled"], {"default": "enabled", "tooltip": "Enable memory optimization to reduce VRAM usage."}),
                "importance_threshold": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001, "tooltip": "Threshold for selective caching. Higher values save more memory but may reduce accuracy."}),
                "max_cache_entries": ("INT", {"default": 1000, "min": 100, "max": 10000, "step": 100, "tooltip": "Maximum number of cache entries to store. Lower values use less memory."}),
                "prune_frequency": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1, "tooltip": "How often to prune the cache (in steps). Lower values save more memory but add overhead."})
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_taylorseer"
    CATEGORY = "TaylorSeer"
    TITLE = "TaylorSeer"
    
    def apply_taylorseer(self, model, model_type: str, fresh_threshold: int, max_order: int, first_enhance: int, last_enhance: int, memory_efficient="enabled", importance_threshold=0.01, max_cache_entries=1000, prune_frequency=5):
        new_model = model.clone()
        if 'transformer_options' not in new_model.model_options:
            new_model.model_options['transformer_options'] = {} 
        
        if "flux" in model_type:
            context = create_fluxpatches_dict(new_model)
        else:
            raise ValueError(f"Unknown type {model_type}")
        
        def unet_wrapper_function(model_function, kwargs):
            input = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]
            # check current step
            sigmas = c["transformer_options"]["sample_sigmas"]
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
                # Configure memory optimization settings
                memory_opt_enabled = (memory_efficient == "enabled")
                new_model.model.diffusion_model.cache_dic, new_model.model.diffusion_model.current = cache_init_flux(fresh_threshold, max_order, first_enhance, last_enhance, len(sigmas))
                
                # Apply memory optimization settings if enabled
                if memory_opt_enabled:
                    new_model.model.diffusion_model.cache_dic['memory_efficient'] = True
                    new_model.model.diffusion_model.cache_dic['importance_threshold'] = importance_threshold
                    new_model.model.diffusion_model.cache_dic['max_cache_entries'] = max_cache_entries
                    new_model.model.diffusion_model.cache_dic['prune_frequency'] = prune_frequency
                else:
                    new_model.model.diffusion_model.cache_dic['memory_efficient'] = False
            with context:
                return model_function(input, timestep, **c)

        new_model.set_model_unet_function_wrapper(unet_wrapper_function)

        return (new_model,)

NODE_CLASS_MAPPINGS = {
    "TaylorSeer": TaylorSeer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TaylorSeer": "TaylorSeer",
}