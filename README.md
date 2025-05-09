# ComfyUI-TaylorSeer

[阅读中文版](./README_zh.md)

This project is the ComfyUI implementation of the TaylorSeer project [https://github.com/Shenyi-Z/TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer).

## Important Note

Please ensure your ComfyUI version is newer than commit `c496e53`.

## Project Updates

- **WIP**: Hidream and WAN 2.1 support is on the way
- **update**: ```2025/04/30```: First release, supporting FLUX.

## Quick Start

### Installation

1. Navigate to `ComfyUI/custom_nodes`
2. Clone this repository
3. Run ComfyUI after installation is complete!

### Running the Workflow

[Reference Workflow for FLUX](./examples/taylorseer_example_flux.json)

## Usage Instructions

Memory Requirements: The cache needs to be stored in GPU memory for efficient computation. For a 1024*1024 image using FLUX FP8 precision model:

- Order 0: Increases VRAM usage by 2GB
- Order 1: Increases VRAM usage by 4GB
- Order 2: Increases VRAM usage by 6GB

VRAM usage increases linearly with resolution and number of images.

Acceleration Ratio: The `first_enhance` parameter can adjust when Taylor Cache intervenes. When first_enhance = 10, with 30 iteration steps, the results are almost lossless compared to the original results, and the acceleration ratio can reach 2x.

## License

The code in this repository is released under the [GNU General Public License v3.0](./LICENSE).
