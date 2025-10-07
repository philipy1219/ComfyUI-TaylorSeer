# ComfyUI-TaylorSeer

[阅读中文版](./README_zh.md)

This project is the ComfyUI implementation of the TaylorSeer project [https://github.com/Shenyi-Z/TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer).

## Important Note

Please ensure your ComfyUI version is newer than commit `c496e53`.

## Project Updates

- **update**: ```2025/10/07```: Based on TaylorSeer-Lite, support Qwwn-Image.
- **update**: ```2025/09/25```: Support [TaylorSeer-Lite](https://github.com/Shenyi-Z/Cache4Diffusion/blob/main/HunyuanImage-2.1/run_hyimage_taylorseer_lite.py), reducing cache quantity with negligible VRAM increase. Based on TaylorSeer-Lite, now supporting WAN 2.1/2.2 models.
- **update**: ```2025/05/25```: Support block swap, now you can run it with low VRAM
- **update**: ```2025/05/13```: Support Hidream, force VRAM purge when checkpoint is changed
- **update**: ```2025/04/30```: First release, supporting FLUX.

## Quick Start

### Installation

1. Navigate to `ComfyUI/custom_nodes`
2. Clone this repository
3. Run ComfyUI after installation is complete!

### Running the Workflow

[Workflow for FLUX](./examples/taylorseer_example_flux.json)

[Workflow for Hidream](./examples/taylorseer_example_hidream_full.json)

[Workflow for FLUX-TaylorSeer-Lite](./examples/taylorseerlite_example_flux.json)

[Workflow for WAN-2.2-TaylorSeer-Lite](./examples/taylorseerlite_example_wan2-2.json)

[Workflow for Qwen-Image-TaylorSeer-Lite](./examples/taylorseerlite_example_qwenimage.json)

## Usage Instructions

### Using TaylorSeer Standard Version

Memory Requirements Flux: The cache needs to be stored in GPU memory for efficient computation. For a 1024*1024 image using FLUX FP8 precision model:

- Order 0: Increases VRAM usage by 2GB
- Order 1: Increases VRAM usage by 4GB
- Order 2: Increases VRAM usage by 6GB

VRAM usage increases linearly with resolution and number of images.

Memory Requirements Hidream: The cache needs to be stored in GPU memory for efficient computation. For a 1024*1024 image using Hidream-full FP8 precision model:

- Order 0: Increases VRAM usage by 5GB
- Order 1: Increases VRAM usage by 10GB
- Order 2: Increases VRAM usage by 15GB

VRAM usage increases linearly with resolution and number of images.

Acceleration Ratio: The `first_enhance` parameter can adjust when Taylor Cache intervenes. When first_enhance = 10, with 30 iteration steps, the results are almost lossless compared to the original results, and the acceleration ratio can reach 2x.

### Using TaylorSeer-Lite

Nearly zero VRAM increase.

**Exciting Performance on WAN 2.2**: For 81 frames generation on RTX 5090, TaylorSeer-Lite achieves remarkable acceleration - **386s vs 1176s** (3.05x speedup) compared to the original implementation!

## Comparison with teacache

Compared to TeaCache, TaylorSeer maintains a higher acceleration ratio and preserves more consistent composition and elements relative to the original image.

| Prompt | Original<br>(steps = 50, 2.42it/s) | TaylorSeer<br>(steps = 50, order = 1, first_enhance = 10, 5.73it/s) | TeaCache<br>(steps = 50, rel_l1_thresh = 0.25, 4.08it/s) | TeaCache<br>(steps = 50, rel_l1_thresh = 0.40, 5.32it/s) |
|--------|----------|------------|-----------|-----------|
| fashion photo of a model wearing black draped plastic fabric designed by Demna Gvasalia, standing in an apocalyptic room, sunglasses, synthetic light, 4K, photoreal, 3D render, | [<img src="./sample_images/12_original.png" width="200px">](./sample_images/12_original.png) | [<img src="./sample_images/12_TaylorSeer.png" width="200px">](./sample_images/12_TaylorSeer.png) | [<img src="./sample_images/12_teacache_25.png" width="200px">](./sample_images/12_teacache_25.png) | [<img src="./sample_images/12_teacache_40.png" width="200px">](./sample_images/12_teacache_40.png) |
| This picture depicts an anime scene that revolves around a young girl. She is characterized by purple hair tied into two ponytails, each decorated with a bow and pink earmuffs. The girl is wearing dark clothes, a vest-style top and shorts. She has a happy expression, a big smile on her face, and a pearl or something delicate in her mouth, as if she is jokingly feeding herself. | [<img src="./sample_images/122_original.png" width="200px">](./sample_images/122_original.png) | [<img src="./sample_images/122_TaylorSeer.png" width="200px">](./sample_images/122_TaylorSeer.png) | [<img src="./sample_images/122_teacache_25.png" width="200px">](./sample_images/122_teacache_25.png) | [<img src="./sample_images/122_teacache_40.png" width="200px">](./sample_images/122_teacache_40.png) |

## License

The code in this repository is released under the [GNU General Public License v3.0](./LICENSE). 
