import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "0"
import numpy as np
import gradio as gr
import supervision as sv
from scipy.ndimage import binary_fill_holes
from ultralytics import YOLOE
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
from gradio_image_prompter import ImagePrompter
import torch

def init_model(model_id, is_pf=False, model_dir="models"):
    filename = f"{model_id}-seg.pt" if not is_pf else f"{model_id}-seg-pf.pt"
    model_path = os.path.join(model_dir, filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在，请下载模型到 {model_dir} 目录")
    model = YOLOE(model_path)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    else:
        print("未检测到GPU，使用CPU运行")
        model.to("cpu")
    return model

@smart_inference_mode()
def yoloe_inference(image, prompts, target_image, model_id, image_size, conf_thresh, iou_thresh, prompt_type, model_dir="models"):
    model = init_model(model_id, model_dir=model_dir)
    kwargs = {}
    if prompt_type == "Text":
        texts = prompts["texts"]
        model.set_classes(texts, model.get_text_pe(texts))
    elif prompt_type == "Visual":
        kwargs = dict(
            prompts=prompts,
            predictor=YOLOEVPSegPredictor
        )
        if target_image:
            model.predict(source=image, imgsz=image_size, conf=conf_thresh, iou=iou_thresh, return_vpe=True, **kwargs)
            model.set_classes(["object0"], model.predictor.vpe)
            model.predictor = None  # unset VPPredictor
            image = target_image
            kwargs = {}
    elif prompt_type == "Prompt-free":
        vocab = model.get_vocab(prompts["texts"])
        model = init_model(model_id, is_pf=True, model_dir=model_dir)
        model2 = init_model(model_id, is_pf=False, model_dir=model_dir)
        model2.eval()
        model2.cuda()
        vocab = model2.get_vocab(prompts["texts"])
        model.set_vocab(vocab, names=prompts["texts"])
        model.model.model[-1].is_fused = True
        model.model.model[-1].conf = 0.001
        model.model.model[-1].max_det = 1000
    results = model.predict(source=image, imgsz=image_size, conf=conf_thresh, iou=iou_thresh, **kwargs)
    detections = sv.Detections.from_ultralytics(results[0])
    resolution_wh = image.size
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(detections['class_name'], detections.confidence)
    ]
    annotated_image = image.copy()
    annotated_image = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.4).annotate(
        scene=annotated_image, detections=detections)
    annotated_image = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=thickness).annotate(
        scene=annotated_image, detections=detections)
    annotated_image = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX, text_scale=text_scale, smart_position=True).annotate(
        scene=annotated_image, detections=detections, labels=labels)
    return annotated_image

def create_app(model_dir="models"):
    def app():
        with gr.Blocks():
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        raw_image = gr.Image(type="pil", label="Image", visible=True, interactive=True)
                        box_image = ImagePrompter(type="pil", label="DrawBox", visible=False, interactive=True)
                        mask_image = gr.ImageEditor(type="pil", label="DrawMask", visible=False, interactive=True, layers=False, canvas_size=(640, 640))
                        target_image = gr.Image(type="pil", label="Target Image", visible=False, interactive=True)
                    
                    yoloe_infer = gr.Button(value="Detect & Segment Objects")
                    prompt_type = gr.Textbox(value="Text", visible=False)
                    with gr.Tab("Text") as text_tab:
                        texts = gr.Textbox(label="Input Texts", value='person,car', placeholder='person,car', visible=True, interactive=True)
                    
                    with gr.Tab("Visual") as visual_tab:
                        with gr.Row():
                            visual_prompt_type = gr.Dropdown(choices=["bboxes", "masks"], value="bboxes", label="Visual Type", interactive=True)
                            visual_usage_type = gr.Radio(choices=["Intra-Image", "Cross-Image"], value="Intra-Image", label="Intra/Cross Image", interactive=True)
                    
                    with gr.Tab("Prompt-Free") as prompt_free_tab:
                        gr.HTML(
                            """
                            <p style='text-align: center'>
                            <b>Prompt-Free Mode</b>
                            </p>
                        """, show_label=False)
                    model_id = gr.Dropdown(
                        label="Model",
                        choices=[
                            "yoloe-v8l",
                        ],
                        value="yoloe-v8l",
                    )
                    image_size = gr.Slider(
                        label="Image Size",
                        minimum=320,
                        maximum=1280,
                        step=32,
                        value=640,
                    )
                    conf_thresh = gr.Slider(
                        label="Confidence Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.25,
                    )
                    iou_thresh = gr.Slider(
                        label="IoU Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.70,
                    )
                with gr.Column():
                    output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
            
            # UI update functions
            def update_text_image_visibility():
                return gr.update(value="Text"), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            def update_visual_image_visiblity(visual_prompt_type, visual_usage_type):
                if visual_prompt_type == "bboxes":
                    return gr.update(value="Visual"), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=(visual_usage_type == "Cross-Image"))
                elif visual_prompt_type == "masks":
                    return gr.update(value="Visual"), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=(visual_usage_type == "Cross-Image"))
            def update_pf_image_visibility():
                return gr.update(value="Prompt-free"), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            def update_visual_prompt_type(visual_prompt_type):
                if visual_prompt_type == "bboxes":
                    return gr.update(visible=True), gr.update(visible=False)
                if visual_prompt_type == "masks":
                    return gr.update(visible=False), gr.update(visible=True)
                return gr.update(visible=False), gr.update(visible=False)
            def update_visual_usage_type(visual_usage_type):
                if visual_usage_type == "Intra-Image":
                    return gr.update(visible=False)
                if visual_usage_type == "Cross-Image":
                    return gr.update(visible=True)
                return gr.update(visible=False)
            
            # Bind UI updates
            text_tab.select(
                fn=update_text_image_visibility,
                inputs=None,
                outputs=[prompt_type, raw_image, box_image, mask_image, target_image]
            )
            visual_tab.select(
                fn=update_visual_image_visiblity,
                inputs=[visual_prompt_type, visual_usage_type],
                outputs=[prompt_type, raw_image, box_image, mask_image, target_image]
            )
            prompt_free_tab.select(
                fn=update_pf_image_visibility,
                inputs=None,
                outputs=[prompt_type, raw_image, box_image, mask_image, target_image]
            )
            visual_prompt_type.change(
                fn=update_visual_prompt_type,
                inputs=[visual_prompt_type],
                outputs=[box_image, mask_image]
            )
            visual_usage_type.change(
                fn=update_visual_usage_type,
                inputs=[visual_usage_type],
                outputs=[target_image]
            )
            
            # Inference function
            def run_inference(raw_image, box_image, mask_image, target_image, texts, model_id, image_size, conf_thresh, iou_thresh, prompt_type, visual_prompt_type, visual_usage_type):
                # Text or Prompt-free
                if prompt_type == "Text" or prompt_type == "Prompt-free":
                    target_image = None
                    image = raw_image
                    if prompt_type == "Prompt-free":
                        ram_tag_path = os.path.join("tools", "ram_tag_list.txt")
                        if os.path.exists(ram_tag_path):
                            with open(ram_tag_path, 'r') as f:
                                texts = [x.strip() for x in f.readlines()]
                        else:
                            gr.Warning("未找到ram_tag_list.txt文件，使用默认类别")
                            texts = ["person", "bus", "car", "dog", "cat"]
                    else:
                        texts = [text.strip() for text in texts.split(',')]
                    prompts = {
                        "texts": texts
                    }
                # Visual prompt
                elif prompt_type == "Visual":
                    if visual_usage_type != "Cross-Image":
                        target_image = None
                    if visual_prompt_type == "bboxes":
                        image, points = box_image["image"], box_image["points"]
                        points = np.array(points)
                        if len(points) == 0:
                            gr.Warning("No boxes are provided. No image output.", visible=True)
                            return gr.update(value=None)
                        bboxes = np.array([p[[0, 1, 3, 4]] for p in points if p[2] == 2])
                        prompts = {
                            "bboxes": bboxes,
                            "cls": np.array([0] * len(bboxes))
                        }
                    elif visual_prompt_type == "masks":
                        image, masks = mask_image["background"], mask_image["layers"][0]
                        masks = np.array(masks.convert("L"))
                        masks = binary_fill_holes(masks).astype(np.uint8)
                        masks[masks > 0] = 1
                        if masks.sum() == 0:
                            gr.Warning("No masks are provided. No image output.", visible=True)
                            return gr.update(value=None)
                        prompts = {
                            "masks": masks[None],
                            "cls": np.array([0])
                        }
                else:
                    raise ValueError(f"Unsupported prompt_type: {prompt_type}")
                return yoloe_inference(image, prompts, target_image, model_id, image_size, conf_thresh, iou_thresh, prompt_type, model_dir)
            
            yoloe_infer.click(
                fn=run_inference,
                inputs=[raw_image, box_image, mask_image, target_image, texts, model_id, image_size, conf_thresh, iou_thresh, prompt_type, visual_prompt_type, visual_usage_type],
                outputs=[output_image],
            )
    return app

def main():
    gradio_app = gr.Blocks()
    with gradio_app:
        logo_html = """
        <h1 style='text-align: center'>
        YOLOE: Real-Time Seeing Anything
        </h1>
        """
        gr.HTML(logo_html)
        with gr.Row():
            with gr.Column():
                app_instance = create_app(model_dir="models")
                app_instance()
    if not os.path.exists("examples"):
        os.makedirs("examples", exist_ok=True)
        print("请将示例图片放入该目录")
    
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
        print("需要的模型文件：yoloe-v8l-seg.pt 等")
    
    if not os.path.exists("tools/ram_tag_list.txt"):
        os.makedirs("tools", exist_ok=True)
        default_tags = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
        with open("tools/ram_tag_list.txt", "w") as f:
            for tag in default_tags:
                f.write(f"{tag}\n")
        print("创建了默认的ram_tag_list.txt文件")
    
    gradio_app.launch(
        server_name="0.0.0.0",
        server_port=80,
        share=False
    )

if __name__ == '__main__':
    main()
