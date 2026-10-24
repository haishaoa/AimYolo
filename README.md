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

## macOS 运行

使用 Python 3.11 环境，在项目根目录执行：

```bash
conda activate AimYolo
python -m pip install torch torchvision
python -m pip install -r requirements.txt
# 验证权重、CPU 推理和 NMS，不访问屏幕或鼠标：
python z_detect5.py --check --device cpu
# 先只检测屏幕并显示结果：
python z_detect5.py --device cpu --view-img --no-mouse
# 启用原有鼠标移动逻辑：
python z_detect5.py --device cpu --view-img
```

macOS 不需要安装 `pywin32`，也不要使用 Windows/NVIDIA 的 `requirements-cuda.txt`。
屏幕截图使用 MSS，鼠标移动使用 pynput，坐标会根据截图尺寸转换到桌面坐标。
当前设备选择器支持 CPU 和 CUDA；Mac 请使用 `--device cpu`。
首次截屏需在“系统设置 → 隐私与安全 → 屏幕与系统音频录制”（旧版为“屏幕录制”）中
允许运行 Python 的应用（例如 VS Code 或终端）；鼠标控制还需要“辅助功能”权限。
修改权限后，按系统提示重启对应应用。在预览窗口按 `q` 或在终端按 `Ctrl+C` 退出。

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
