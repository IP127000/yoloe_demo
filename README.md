# yoloe_demo
a demo of yoloe, including gradio app demo , yoloe inference and train

## Usage
```bash
git clone https://github.com/THU-MIG/yoloe.git
cd yoloe
pip install -r requirements.txt
wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt
pip install gradio==4.42.0 gradio_image_prompter==0.1.0 fastapi==0.112.2 huggingface-hub==0.26.3 gradio_client==1.3.0 pydantic==2.10.6
python app.py
# visit http://127.0.0.1:80
```
