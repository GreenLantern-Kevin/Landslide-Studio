# LandslideStudio

面向滑坡遥感研究的本地分析工作台，集成遥感影像合成、滑坡语义分割、视频光流分析，以及基于上海交通大学致远一号AI模型API的辅助分析与报告生成。平台采用 FastAPI 后端和原生 HTML/CSS/JavaScript 前端，在浏览器中统一管理输入、推理任务和结果。

仓库包含应用源码、推理适配器、示例配置与 Windows 安装脚本；基础模型、LoRA、分割权重、API 密钥、数据集和个人运行结果均需自行准备。

## 功能概览

| 模块 | 能力 | 主要输出 |
| --- | --- | --- |
| 遥感影像合成 | Stable Diffusion 1.5、Stable Diffusion 2.1 768、SDXL；按基础模型加载滑坡或非滑坡 LoRA，自动补充激活词 | 合成影像、生成参数 |
| 滑坡语义分割 | M2F-DeepLab 推理；支持 RGB+DEM 和缺失 DEM 的 RGB-only 路径；自动读取 checkpoint 参数 | 掩膜、概率、叠加图、区域统计；有地理参考时导出 GeoTIFF |
| 滑坡视频分析 | Lucas–Kanade 稀疏光流、Farnebäck 稠密光流；ROI、背景特征防抖、轨迹和任意帧速度分布可视化 | 分析视频、逐帧记录、速度分布图及叠加图 |
| 辅助分析与报告生成 | 视觉解读、文本证据报告、本地证据摘要；可关联分割或视频结果 | 区分计算证据、模型观察、推断与未知项的报告 |

四个模块支持任务命名、历史结果查看、重命名与文件下载。图像生成、分割和光流计算在本机执行；辅助分析按所选模式调用配置的 API。

## 快速开始

适用于 Windows 与 Python 3.10，安装 Python 时应包含 Windows Python Launcher。

1. 下载或克隆仓库，运行 `01_安装基础环境.cmd`，安装界面、API 与光流依赖。
2. 如需影像合成或分割，运行 `02_安装AI环境.cmd`，安装 PyTorch 2.7.1 / torchvision 0.22.1 的 CUDA 12.6 版本，并使用兼容的 NVIDIA 驱动。无可用 NVIDIA GPU 时选择 `02b_安装CPU推理环境.cmd`；运行速度取决于模型与硬件。
3. 将 `.env.example` 复制为 `.env`、`config.example.json` 复制为 `config.local.json`，按所需模块准备模型和配置。
4. 运行 `00_启动平台.cmd`，打开 [本机工作台](http://127.0.0.1:8765)。通过“模型与配置”页面或 `04_环境自检.cmd` 查看依赖和文件状态。
5. 结束使用时运行 `03_停止平台.cmd`。关闭浏览器标签页不会停止后台服务。

各模块按需启用；使用界面和光流分析无需先下载全部模型。更换电脑或虚拟环境失效时，运行 `06_迁移后重建环境.cmd` 重建 `.venv`，并核对配置中的自定义路径。

## 模型与配置

```text
models/
  base/
    sd15/       # 完整 Diffusers 模型目录
    sd21/       # Stable Diffusion 2.1 768
    sdxl/       # SDXL Base 1.0
  lora/
    sd15/
    sd21/
    sdxl/
  segmentation/
    m2f_deeplab.pth
```

**影像合成。** 运行 `07_下载或续传基础模型.cmd` 下载基础模型；来源及固定版本见 [模型来源配置](scripts/model_sources.json)。也可使用 `.venv\Scripts\python.exe scripts/download_base.py sdxl` 单独下载。基础模型必须为包含 `model_index.json` 及所需组件的完整 Diffusers 目录。

在每个 `models/lora/<基础模型>/` 目录中放置与该基础模型及版本匹配的 LoRA：

| 类别 | 文件名 | 自动激活词 |
| --- | --- | --- |
| 滑坡 | `Southwest_China_Landslide.safetensors` | `Southwest_China_Landslide` |
| 非滑坡 | `Southwest_China_Terrain.safetensors` | `Southwest_China_Terrain` |

LoRA 不可跨基础模型直接混用。下载脚本仅下载基础模型，不提供个人 LoRA 或分割权重。

**语义分割。** 将训练完成的权重放入 `models/segmentation/m2f_deeplab.pth`，推荐保留训练配置元数据的完整 checkpoint。默认使用 M2F-DeepLab 完整 E 模型 `m2f_deeplabv3plus_xception16`，两类输出；优先匹配 checkpoint 记录，未记录输出步长时使用配置默认值 16。自动推理读取 `eval_resize` / `val_size`，缺省为 256，记录为 0 时使用原图尺寸；训练裁剪尺寸不替代评估尺寸。

DEM 应为与 RGB 逐像元配准、编码方式与训练一致的 `uint8` 三通道高程/坡度/坡向地形图，两种输入均使用 ImageNet 标准化。仅上传 RGB 时，完整模型通过显式 `has_dem=False` 使用 FHM 特征补全。架构参数、权重兼容边界和结果编码见 [分割推理说明](docs/分割推理.md)。

**辅助分析。** 在 `.env` 中填写 `SJTU_API_KEY`，在 `config.local.json` 的 `llm` 中配置上海交通大学致远一号AI模型API地址、账户可用的模型 ID 及 `supports_vision`。视觉解读需要可用的视觉模型和本次图片发送确认；文本模式仅处理证据与备注。文本报告和本地摘要均明确说明输入依据与是否调用模型。示例模型 ID 不代表账户具有对应权限。

## 文档与架构

- [使用指南](docs/使用指南.md)：安装配置、四个模块的操作流程、结果管理与排错。平台内可通过“打开逐步使用指南”阅读。
- [分割推理说明](docs/分割推理.md)：M2F-DeepLab 版本、输入约定、参数优先级、严格权重加载及结果解释。

| 目录或文件 | 职责 |
| --- | --- |
| `app/` | FastAPI 服务、配置与任务管理 |
| `app/services/` | 影像生成、分割、光流及报告适配器 |
| `dist/` | 原生浏览器界面，无需额外前端构建 |
| `vendor/M2F-DeepLab/network/` | 固定版本的分割网络定义，仅用于推理 |
| `scripts/` | 安装、启停、自检、下载和权重检查工具 |
| `docs/` | 操作与推理说明；HTML 指南由 Markdown 生成 |
| `config.example.json`、`.env.example` | 不含个人密钥的配置模板 |

## 算法来源

- [M2F-DeepLab](https://github.com/GreenLantern-Kevin/M2F-DeepLab)：分割网络固定于提交 `e655ee310f5e9dfafceb33ca2afc92f214e14c29`，本仓库仅接入推理所需网络定义，不运行训练流程。来源记录见 [UPSTREAM.md](vendor/M2F-DeepLab/UPSTREAM.md)。
- [Optical-Flow-Seismic-Rock-Avalanche-Detection](https://github.com/GreenLantern-Kevin/Optical-Flow-Seismic-Rock-Avalanche-Detection)：光流研究代码，平台提供适用于后台任务和浏览器结果展示的改编实现。
- 影像合成使用 Hugging Face Diffusers 加载 Stable Diffusion 系列模型与 LoRA。

保留第三方源码中的署名与声明；代码和模型分别遵循其上游许可及使用条件。

## 验证范围与使用边界

已使用真实训练权重验证 RGB+DEM 与 RGB-only 两条 GPU 推理路径；这不构成正式精度评测，也不保证任意 checkpoint 兼容。加载采用 `weights_only=True`，校验全部推理参数并以 `strict=True` 加载；仅对已确认的 schema 2 训练辅助头提供有限兼容。

分割像素占比是候选区域统计，概率未经校准；无真值标签时不报告准确率、IoU 或 F1。光流速度以处理分辨率下的 px/s 表示，缺少空间尺度和相机几何标定时不能解释为 m/s。模型结果用于研究与辅助判读，需结合原始资料和独立核验使用。

服务默认面向本机运行，未提供公网账户认证或多用户权限管理。运行数据、缓存与日志分别保存在 `data/jobs/`、`.cache/` 和 `logs/`；清理前应停止服务并备份所需结果。
