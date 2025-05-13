# ComfyUI-TaylorSeer

[Read English Version](./README.md)

本项目是TaylorSeer项目的ComfyUI实现 [https://github.com/Shenyi-Z/TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer)。

## 重要提示

请确保您的ComfyUI版本新于提交`c496e53`。

## 项目更新

- **进行中**: WAN 2.1支持即将到来
- **更新**: ```2025/05/13```: 支持Hidream，当checkpoint改变时，强制卸载旧模型显存
- **更新**: ```2025/04/30```: 首次发布，支持FLUX。

## 快速开始

### 安装

1. 导航到`ComfyUI/custom_nodes`
2. 克隆此仓库
3. 安装完成后运行ComfyUI！

### 运行工作流

[FLUX参考工作流](./workflows/taylorseer_example_flux.json)

[Hidream参考工作流](./workflows/taylorseer_example_hidream_full.json)

## 使用说明

FLUX内存需求：缓存需要存储在GPU内存中以进行高效计算。对于使用FLUX FP8精度模型的1024*1024图像：

- 阶数0：增加VRAM使用量2GB
- 阶数1：增加VRAM使用量4GB
- 阶数2：增加VRAM使用量6GB

VRAM使用量随分辨率和图像数量线性增加。

Hidream内存需求：缓存需要存储在GPU内存中以进行高效计算。对于使用Hidream-full FP8精度模型的1024*1024图像：

- 阶数0：增加VRAM使用量5GB
- 阶数1：增加VRAM使用量10GB
- 阶数2：增加VRAM使用量15GB

VRAM使用量随分辨率和图像数量线性增加。

加速比例：`first_enhance`参数可以调整Taylor Cache介入的时间。当first_enhance = 10时，在30次迭代步骤下，结果与原始结果几乎无损，加速比可达2倍。

## 许可证

本仓库中的代码在[GNU通用公共许可证v3.0](./LICENSE)下发布。