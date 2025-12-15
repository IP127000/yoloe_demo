**Read this in other languages: [简体中文](README.zh.md)
# yoloe_demo
a demo of yoloe, including gradio app demo , yoloe inference and train
## Capabilities
Unlike traditional CV detection models, YOLOE is capable of abstract semantic detection, such as identifying a "person who needs help." It can be applied in security surveillance and home monitoring scenarios

![person who need help](images/example_1.webp){:width="50%" height="50%"}

## Usage
```bash
git clone https://github.com/THU-MIG/yoloe.git
cd yoloe
pip install -r requirements.txt
wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt
pip install gradio==4.42.0 gradio_image_prompter==0.1.0 fastapi==0.112.2 huggingface-hub==0.26.3 gradio_client==1.3.0 pydantic==2.10.6
python yoloe_gradio_demo.py
# visit http://127.0.0.1:80
```
If you encounter pip-related issues, comment out "import pip" in file "yoloe/third_party/lvis-api/setup.py"
