# M2F-DeepLab 推理网络

来源：https://github.com/GreenLantern-Kevin/M2F-DeepLab

固定提交：`e655ee310f5e9dfafceb33ca2afc92f214e14c29`。

本目录仅包含平台推理需要的原始 `network/` 定义，不包含训练数据、权重或训练入口。
网络源码保持该提交原样；平台的输入预处理、checkpoint 兼容加载和任务接口位于 `app/services/segmentation.py` 与 `app/checkpoints.py`。

第三方代码来源和许可声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
