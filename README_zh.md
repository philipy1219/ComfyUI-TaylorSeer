# ComfyUI-TaylorSeer

[Read this in English](./README.md)

本项目为TaylorSeer项目[https://github.com/Shenyi-Z/TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer)在ComfyUI上的实现。

## 重要提示

请确保您使用的ComfyUI版本高于`c496e53`。

## 项目更新

- **update**: ```2025/04/30```: 首次发布，支持FLUX。

## 快速开始

### 安装

1. 前往 `ComfyUI/custom_nodes`  
2. 克隆此仓库
3. 安装完成后运行 ComfyUI！  

### 运行工作流

[参考工作流FLUX](./workflows/taylorseer_example_flux.json)

## 使用说明

显存要求：缓存需要存储到显存进行高效计算，1张1024*1024尺寸图像，使用FLUX FP8精度模型，阶数为0时增加显存占用2G，阶数为1时增加显存占用4G，阶数为2时增加显存占用6G，显存占用随分辨率和张数的增加线性增加。

加速比：first_enhance参数可以调节Taylor Cache介入的时机，first_enhance = 10时，30个step的迭代结果，和原始结果对比几乎无损，加速比可以达到2。

## 模型协议

本仓库代码使用 [GNU General Public License v3.0 协议](./LICENSE) 发布。
