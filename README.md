# ComfyUI-TaylorSeer

[阅读中文版](./README_zh.md)

This project is the ComfyUI implementation of the TaylorSeer project [https://github.com/Shenyi-Z/TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer).

## Important Note

Please ensure your ComfyUI version is newer than commit `c496e53`.

## Project Updates

- **update**: ```2025/04/30```: First release, supporting FLUX.

## Quick Start

### Installation

1. Navigate to `ComfyUI/custom_nodes`
2. Clone this repository
3. Run ComfyUI after installation is complete!

### Running the Workflow

[Reference Workflow for FLUX](./workflows/taylorseer_example_flux.json)

## Usage Instructions

### Memory Requirements
The cache needs to be stored in GPU memory for efficient computation. For a 1024*1024 image using FLUX FP8 precision model with standard settings:
- Order 0: Increases VRAM usage by 2GB
- Order 1: Increases VRAM usage by 4GB
- Order 2: Increases VRAM usage by 6GB

With memory optimization enabled (default), VRAM usage can be reduced by 30-50% depending on settings.

### Memory Optimization Settings
TaylorSeer now includes memory optimization features to reduce VRAM usage:

- **Memory Efficient**: Enable/disable memory optimization (enabled by default)
- **Importance Threshold**: Controls selective caching (0.01 default). Higher values save more memory but may slightly reduce accuracy.
- **Max Cache Entries**: Limits the total number of cache entries (1000 default). Lower values use less memory.
- **Prune Frequency**: How often to remove least important cache entries (every 5 steps by default).

### Performance
Acceleration Ratio: The `first_enhance` parameter can adjust when Taylor Cache intervenes. When first_enhance = 10, with 30 iteration steps, the results are almost lossless compared to the original results, and the acceleration ratio can reach 2x.

## License

The code in this repository is released under the [GNU General Public License v3.0](./LICENSE).
