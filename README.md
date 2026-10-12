# LandslideStudio

滑坡影像分析平台：集成遥感影像合成、多模态语义分割、光流视频分析，以及基于上海交通大学致远一号AI模型API的辅助分析与报告生成。采用 FastAPI 后端与 HTML/CSS/JavaScript 前端，在本机浏览器中运行。

本仓库提供部署所需源码、示例配置及安装启动脚本，不包含基础模型、LoRA、分割权重、API Key、数据集及个人运行结果。

## 功能

- **遥感影像合成**：支持 Stable Diffusion 1.5、Stable Diffusion 2.1、SDXL；按模型与类别加载 LoRA，自动补充滑坡/非滑坡激活词。
- **滑坡语义分割**：调用 M2FDeepLabV3Plus 推理网络，支持 RGB+DEM 和 RGB-only 输入，输出掩膜、叠加图及统计结果。
- **滑坡视频分析**：稀疏/稠密光流、轨迹可视化、视频与记录数据导出、任意帧速度分布图及叠加图。
- **辅助分析与报告生成**：通过配置的视觉/文本模型API生成报告；未调用大模型时明确标识为本地证据摘要。
- **任务与结果**：可选任务命名、历史任务重命名、结果查看与下载。

## 快速开始（Windows）

1. 安装 Python 3.10（包含 Windows Python Launcher），下载或克隆本仓库并解压到本地目录。
2. 双击 `01_安装基础环境.cmd`，安装界面、API和光流依赖。
3. 需要影像生成或语义分割时，双击 `02_安装AI环境.cmd`。当前脚本安装 PyTorch 2.7.1 / torchvision 0.22.1 的 CUDA 12.6 版本，需要兼容的 NVIDIA 显卡驱动。无可用 NVIDIA GPU 时可选择 `02b_安装CPU推理环境.cmd`，实际推理速度取决于硬件。
4. 按下节放置模型，并将 `.env.example` 复制为 `.env`、`config.example.json` 复制为 `config.local.json`，填写自己的配置。
5. 双击 `00_启动平台.cmd`，在浏览器打开 `http://127.0.0.1:8765`。可用 `04_环境自检.cmd` 检查环境及文件状态。
6. 结束使用时运行 `03_停止平台.cmd`。只关闭浏览器标签页不会停止后台服务。

只查看界面或使用光流分析，无需先下载所有模型。不同功能按自己的依赖与模型配置启用。服务默认仅在本机使用，不包含面向公网的账户认证或多用户权限管理。

## 模型与配置

```text
models/
  base/
    sd15/       # 完整 Diffusers 格式基础模型
    sd21/       # 当前下载脚本选用 SD 2.1 768 版本
    sdxl/       # SDXL Base 1.0
  lora/
    sd15/
    sd21/
    sdxl/
  segmentation/
    rgb_dem.pth
```

可运行 `07_下载或续传基础模型.cmd` 下载三个基础模型，支持续传；文件来源和固定版本记录在 `scripts/model_sources.json`。也可在安装环境后单独下载：

```powershell
.venv\Scripts\python.exe scripts/download_base.py sdxl
```

模型使用需遵循各自上游许可及使用条件。下载脚本不提供、也不会训练个人 LoRA 和分割权重。

在每个 `models/lora/<基础模型>/` 目录中分别放置与该基础模型匹配的：

| 类别 | 文件名 | 自动添加的激活词 |
| --- | --- | --- |
| 滑坡 | `Southwest_China_Landslide.safetensors` | `Southwest_China_Landslide` |
| 非滑坡 | `Southwest_China_Terrain.safetensors` | `Southwest_China_Terrain` |

LoRA 不能在不同基础模型之间直接混用；SD 2.1 的版本与训练配置也应一致。基础模型目录需要包含 `model_index.json` 及对应组件，不能只放一个权重文件。

将双模态分割权重放在 `models/segmentation/rgb_dem.pth`。默认网络为 `m2f_deeplabv3plus_xception16`、`output_stride=16`、类别数 2；以实际训练配置为准。平台会读取可获取的 checkpoint 元数据，未记录的参数仍需人工核对。DEM 输入是预先制作且与 RGB 配准的三通道高程、坡度、坡向伪彩色图，不应直接替换成任意灰度图。RGB-only 使用缺失 DEM 的配置策略，并不意味着双模态权重变成了独立训练的 RGB 模型。

在 `.env` 中设置自己的 `SJTU_API_KEY`。在 `config.local.json` 的 `llm` 配置中填写服务地址、账户可用的模型ID，并按模型实际能力设置 `supports_vision`。示例模型ID不保证对所有账户有效；视觉分析需选用支持图像输入的模型。不要提交 `.env` 或个人配置文件。

## 使用与结果

四个功能页面均提供“任务名称（可选）”；未填写时保留任务类型名称，填写后显示为“任务类型-任务名称”。历史任务可在“任务与结果”中重命名。

运行结果保存在 `data/jobs/<任务ID>/`，缓存位于 `.cache/`，日志位于 `logs/`。视频分析及逐点记录可能占用较大磁盘空间。需要清理时，先停止服务，备份有用结果后再删除不需要的任务目录；删除任务目录会同时移除其结果。

光流速度默认依据图像像素位移和帧率计算。只有提供适当的空间尺度标定时才能解释为物理速度；单一尺度不能消除透视造成的全部误差。模型输出用于研究和辅助判读，不等同于经现场核验的灾害结论。

移动目录后应重新启动服务；更换电脑或虚拟环境失效时使用 `06_迁移后重建环境.cmd` 重建依赖，并检查个人配置中的绝对路径。不要把另一台电脑的 `.venv` 当作通用可移植环境。

## 目录结构

| 路径 | 用途 |
| --- | --- |
| `app/` | FastAPI 服务、任务管理和四个功能适配器 |
| `dist/` | 浏览器界面，无需额外前端构建步骤 |
| `scripts/` | 安装、启停、自检、下载和权重检查工具 |
| `vendor/M2FDeepLabV3Plus/network/` | 分割推理所需的原始网络定义 |
| `docs/使用指南.html` | 平台内可打开的简明使用指南 |
| `config.example.json`、`.env.example` | 无个人密钥的示例配置 |

## 算法来源

- [M2FDeepLabV3Plus](https://github.com/GreenLantern-Kevin/M2FDeepLabV3Plus)：分割网络；本仓库仅附推理所需网络定义，版本见 `vendor/M2FDeepLabV3Plus/UPSTREAM.md`。
- [Optical-Flow-Seismic-Rock-Avalanche-Detection](https://github.com/GreenLantern-Kevin/Optical-Flow-Seismic-Rock-Avalanche-Detection)：光流研究代码；平台包含适合后台任务执行的改编实现。
- 影像生成通过 Hugging Face Diffusers 加载 Stable Diffusion 系列模型与 LoRA。

仓库发布不改变第三方代码或模型各自的许可条件。
