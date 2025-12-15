**Read this in other languages：[English](README.md)**
# yoloe_demo
YOLOE 示例项目，包含 Gradio 交互界面演示、YOLOE 推理及训练功能
## 功能特点
与传统 CV 检测模型不同，YOLOE 能够进行抽象语义检测，识别例如:person who need help。
可应用于安防监控与家庭看护等场景。

<img src="images/example_1.webp" width="50%" alt="person who need help">

## 使用方式
```bash
git clone https://github.com/THU-MIG/yoloe.git
cd yoloe
pip install -r requirements.txt
wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt
pip install gradio==4.42.0 gradio_image_prompter==0.1.0 fastapi==0.112.2 huggingface-hub==0.26.3 gradio_client==1.3.0 pydantic==2.10.6
python yoloe_gradio_demo.py
# 访问 http://127.0.0.1:80
```
若遇到 pip 相关问题，请注释掉文件 "yoloe/third_party/lvis-api/setup.py" 中的 "import pip"
