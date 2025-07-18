import os
from flask import Flask, render_template, request
from PIL import Image
import torch
import torch.nn as nn
import timm
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
from qwen_vl_utils import process_vision_info
from peft import PeftModel

# === Flask Setup ===
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load TinyViT Classifier ===
model = timm.create_model("tiny_vit_21m_224.dist_in22k", pretrained=True)
model.head.fc = nn.Linear(model.head.fc.in_features, 1)
model.load_state_dict(torch.load("tiny_vit_modified.pth", map_location=device))
model.to(device)
model.eval()
data_config = timm.data.resolve_model_data_config(model)
test_transforms = timm.data.create_transform(**data_config, is_training=False)

# === Load Qwen2-VL-2B-Instruct ===
model_name = "Qwen/Qwen2-VL-2B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model2 = AutoModelForVision2Seq.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)
model2 = PeftModel.from_pretrained(model2, "/Users/ajaypillai/Desktop/venv/flask_project /qwen_modelnewfine")
model2.eval()

# === Qwen Prompt ===

qwen_prompt = """ Analyze the provided image ,a model trained to detect fake visuals detected it as FAKE . Identify and explain artifacts that indicate it is fake. Focus primarily on the original image to identify and explain distinguishing artifacts that indicate it is fake. Use the Grad-CAM output for reference only when necessary. Provide clear, concise explanations (maximum 50 words each) using the specified artifacts below. Include positional references like 'top left' or 'bottom right' when relevant. DO NOT include any other sentences or artifacts in your response. Select only 6-7 relevant artifacts.
make prediction of the category of the image ,verify from the artifacts below 
categories = {
    "airplane": [
        "Artificial noise patterns in uniform surfaces",
        "Metallic surface artifacts",
        "Impossible mechanical connections",
        "Inconsistent scale of mechanical parts",
        "Physically impossible structural elements",
        "Implausible aerodynamic structures",
        "Misaligned body panels",
        "Impossible mechanical joints",
        "Distorted window reflections",
    ],
    "automobile": [
        "Artificial noise patterns in uniform surfaces",
        "Metallic surface artifacts",
        "Impossible mechanical connections",
        "Inconsistent scale of mechanical parts",
        "Physically impossible structural elements",
        "Incorrect wheel geometry",
        "Misaligned body panels",
        "Impossible mechanical joints",
        "Distorted window reflections",
    ],
    "ship": [
        "Artificial noise patterns in uniform surfaces",
        "Metallic surface artifacts",
        "Impossible mechanical connections",
        "Inconsistent scale of mechanical parts",
        "Physically impossible structural elements",
        "Misaligned body panels",
    ],
    "truck": [
        "Artificial noise patterns in uniform surfaces",
        "Metallic surface artifacts",
        "Impossible mechanical connections",
        "Inconsistent scale of mechanical parts",
        "Physically impossible structural elements",
        "Incorrect wheel geometry",
        "Misaligned body panels",
        "Impossible mechanical joints",
        "Distorted window reflections",
    ],
    "bird": [
        "Unrealistic eye reflections",
        "Misshapen ears or appendages",
        "Anatomically impossible joint configurations",
        "Unnatural pose artifacts",
        "Biological asymmetry errors",
        "Regular grid-like artifacts in textures",
        "Impossible foreshortening in animal bodies",
        "Misaligned bilateral elements in animal faces",
        "Over-smoothing of natural textures",
    ],
    "cat": [
        "Unrealistic eye reflections",
        "Misshapen ears or appendages",
        "Anatomically impossible joint configurations",
        "Unnatural pose artifacts",
        "Biological asymmetry errors",
        "Regular grid-like artifacts in textures",
        "Impossible foreshortening in animal bodies",
        "Misaligned bilateral elements in animal faces",
        "Over-smoothing of natural textures",
        "Anatomically incorrect paw structures",
        "Improper fur direction flows",
    ],
    "deer": [
        "Unrealistic eye reflections",
        "Misshapen ears or appendages",
        "Anatomically impossible joint configurations",
        "Unnatural pose artifacts",
        "Biological asymmetry errors",
        "Regular grid-like artifacts in textures",
        "Impossible foreshortening in animal bodies",
        "Misaligned bilateral elements in animal faces",
        "Over-smoothing of natural textures",
        "Improper fur direction flows",
    ],
    "dog": [
        "Unrealistic eye reflections",
        "Misshapen ears or appendages",
        "Anatomically impossible joint configurations",
        "Unnatural pose artifacts",
        "Biological asymmetry errors",
        "Regular grid-like artifacts in textures",
        "Impossible foreshortening in animal bodies",
        "Misaligned bilateral elements in animal faces",
        "Over-smoothing of natural textures",
        "Dental anomalies in mammals",
        "Anatomically incorrect paw structures",
        "Improper fur direction flows",
    ],
    "frog": [
        "Unrealistic eye reflections",
        "Misshapen ears or appendages",
        "Anatomically impossible joint configurations",
        "Unnatural pose artifacts",
        "Biological asymmetry errors",
        "Regular grid-like artifacts in textures",
        "Impossible foreshortening in animal bodies",
        "Misaligned bilateral elements in animal faces",
        "Over-smoothing of natural textures",
    ],
    "horse": [
        "Unrealistic eye reflections",
        "Misshapen ears or appendages",
        "Anatomically impossible joint configurations",
        "Unnatural pose artifacts",
        "Biological asymmetry errors",
        "Regular grid-like artifacts in textures",
        "Impossible foreshortening in animal bodies",
        "Misaligned bilateral elements in animal faces",
        "Over-smoothing of natural textures",
        "Dental anomalies in mammals",
    ],
    "major": [
        "Discontinuous surfaces",
        "Non-manifold geometries in rigid structures",
        "Asymmetric features in naturally symmetric objects",
        "Texture bleeding between adjacent regions",
        "Excessive sharpness in certain image regions",
        "Artificial smoothness",
        "Movie-poster-like composition of ordinary scenes",
        "Unnatural lighting gradients",
        "Fake depth of field",
        "Abruptly cut-off objects",
        "Color coherence breaks",
        "Spatial relationship errors",
        "Depth perception anomalies",
        "Over-sharpening artifacts",
        "Incorrect reflection mapping",
        "Inconsistent object boundaries",
        "Floating or disconnected components",
        "Texture repetition patterns",
        "Unrealistic specular highlights",
        "Inconsistent material properties",
        "Inconsistent shadow directions",
        "Multiple light source conflicts",
        "Missing ambient occlusion",
        "Incorrect perspective rendering",
        "Scale inconsistencies within single objects",
        "Aliasing along high-contrast edges",
        "Blurred boundaries in fine details",
        "Jagged edges in curved structures",
        "Random noise patterns in detailed areas",
        "Loss of fine detail in complex structures",
        "Artificial enhancement artifacts",
        "Repeated element patterns",
        "Systematic color distribution anomalies",
        "Frequency domain signatures",
        "Unnatural color transitions",
        "Resolution inconsistencies within regions",
        "Glow or light bleed around object boundaries",
        "Ghosting effects: Semi-transparent duplicates of elements",
        "Cinematization effects",
        "Dramatic lighting that defies natural physics",
        "Artificial depth of field in object presentation",
        "Unnaturally glossy surfaces",
        "Synthetic material appearance",
        "Multiple inconsistent shadow sources",
        "Exaggerated characteristic features",
        "Scale inconsistencies within the same object class",
        "Incorrect skin tones",
    ],
}
Output Format:
Write each artifact and explanation on a separate line, using the format:
Artifact Name: Explanation.
For example:

Notes:
Explanations should remain under 50 words for clarity.
AVOID referencing artifacts not listed or including extra commentary.

Choose from the below list depending on the image:

"""  

# === Convert to Qwen Structured Message Format ===
def convert_to_structured_messages(messages, image_paths):
    structured_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            parts = content.split("<image>")
            content_list = []
            for i in range(content.count("<image>")):
                content_list.append({"type": "image", "image": f"file://{image_paths[i]}"})
            remaining_text = parts[-1].strip()
            if remaining_text:
                content_list.append({"type": "text", "text": remaining_text})
            structured_messages.append({"role": "user", "content": content_list})
        else:
            structured_messages.append({"role": role, "content": msg["content"]})
    return structured_messages

# === Qwen Inference Function ===
def qwen_infer(image_path):
    entry = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant, who can classify real and fake images and clearly produce different outputs instead of just copying the input."
            },
            {
                "role": "user",
                "content": f"<image>\n{qwen_prompt}"
            }
        ],
        "images": [image_path]
    }

    messages = convert_to_structured_messages(entry["messages"], entry["images"])
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["input_ids"] = inputs["input_ids"].long()
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].long()

    with torch.no_grad():
        generated_ids = model2.generate(**inputs, max_new_tokens=256)

    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    output = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output[0]

# === Flask Route ===
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    explanation = None
    image_path = None

    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            file.stream.flush()

            try:
                img = Image.open(filepath).convert("RGB")
                input_tensor = test_transforms(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor).view(-1, 1)
                    prob = torch.sigmoid(output)
                    predicted = (prob > 0.5).float().item()

                if predicted == 0.0:
                    result = "FAKE"
                    explanation = qwen_infer(filepath)
                else:
                    result = "REAL"
                    explanation = "This image was classified as real. No suspicious artifacts detected."

                image_path = filepath

            except Exception as e:
                return render_template("index.html", error=f"Error: {str(e)}")

    return render_template("index.html", image_path=image_path, result=result, explanation=explanation)

# === Run Flask App ===
if __name__ == '__main__':
    app.run(debug=True)