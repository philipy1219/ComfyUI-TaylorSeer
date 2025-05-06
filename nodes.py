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
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_taylorseer"
    CATEGORY = "TaylorSeer"
    TITLE = "TaylorSeer"
    
    def apply_taylorseer(self, model, model_type: str, fresh_threshold: int, max_order: int, first_enhance: int, last_enhance: int):
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
                new_model.model.diffusion_model.cache_dic, new_model.model.diffusion_model.current = cache_init_flux(fresh_threshold, max_order, first_enhance, last_enhance, len(sigmas))
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