# AimYolo
AI外挂——基于YOLOv5的射击类游戏瞄准辅助。An AI plug-in - targeting aid for shooting games based on YOLOv5.

---
## 详细内容见博客
https://e-cho.top/2023/01/30/2.aimyolo/  

https://wsblog.netlify.app/2023/01/30/2.aimyolo/

---
## 效果图
![image](https://e-cho.top/medias/BlogPictures/1.%20AimYolo/cover/cover.PNG)

---
## 内容更新
2023.4.8：更新了requirements.txt中的内容，将pynput前的注释符号删掉。



## RTX 5070 Ti / Windows 环境

已验证：Python 3.11、PyTorch 2.14.1+cu132、torchvision 0.29.1+cu132、NumPy 2.4.6、NVIDIA 驱动 617.42。
保留本地 YOLOv5 2.0 模型结构和现有权重，不通过 ultralytics 包加载旧权重。

```powershell
conda activate AimYolo
# 当前 AimYolo 环境的 GPU PyTorch 已可用，无需重新安装。
python -m pip install -r requirements.txt
# 不截屏、不移动鼠标，验证权重、GPU FP16 推理和 CUDA NMS：
python z_detect5.py --check --device 0
# 启动屏幕检测（会按原有逻辑移动鼠标），Ctrl+C 退出：
python z_detect5.py --device 0 --view-img
# CPU 兼容性检查：
python z_detect5.py --check --device cpu
```

新建环境时，先安装 `requirements-cuda.txt`，再安装 `requirements.txt`。
训练附加依赖在 `requirements-train.txt`；导出附加依赖在 `requirements-export.txt`。
训练数据需自行准备：当前仓库没有 `data/mydata/images/train/`，不能直接开始完整训练。

旧版 `.pt` 保存了完整 Python 模型，兼容加载器使用 `weights_only=False`，仅加载可信权重和数据缓存。
截图区域自动匹配主显示器分辨率。`--weights` 可以指定本地权重，`--device cpu` 可以禁用 GPU。
