# Fake Image Classifier 🔍🧠

This project implements a **multi-stage pipeline** for detecting fake images and explaining the visual artifacts that suggest manipulation. It combines state-of-the-art models in computer vision and natural language processing to deliver both **classification** and **explanation**.

---

## 🧬 Pipeline Overview

1. **Fake/Real Classification**
   - A **Tiny Vision Transformer (TinyViT)** is trained to classify images as either **real** or **fake**.

2. **Resolution Enhancement**
   - Images predicted as fake are passed through **ESRGAN** (Enhanced Super-Resolution Generative Adversarial Network) to enhance resolution and reveal finer details.

3. **Artifact Explanation**
   - The enhanced fake images are analyzed using a **pretrained Qwen Vision-Language model**, which generates natural language explanations of visual artifacts that led to the "fake" classification.

---

## 🛠️ Technologies Used

- **TinyViT** – Lightweight Vision Transformer for image classification.
- **ESRGAN** – Deep learning model for image super-resolution.
- **Qwen-VL** – Vision-Language model for generating human-readable explanations.
- **PyTorch** – Deep learning framework.


---

## 🧪 Example Workflow

1. Input Image:
   ![input](assets/sample_input.jpg)

2. Output:
   - **Prediction:** Fake
   - **Enhanced Image:** ![enhanced](assets/sample_enhanced.jpg)
   - **Explanation:**  
     > "The facial shadows and lighting are inconsistent. Hair strands appear unnaturally smoothed, especially around the edges—typical signs of AI-generated content."

---


