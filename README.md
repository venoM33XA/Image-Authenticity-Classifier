# Image_Authenticity_Classifier 🔍🧠

This project implements a **multi-stage pipeline** for detecting fake images and explaining the visual artifacts that suggest manipulation. It combines state-of-the-art models in computer vision and natural language processing to deliver both **classification** and **explanation**.

---

## 🧬 Pipeline Overview

1. **Fake/Real Classification**
   - A **Tiny Vision Transformer (TinyViT)** is trained to classify images as either **real** or **fake**.
   - Trained on images with perturbations -> Projected Gradient descent
   -  Gave >99% accuracy on train ,as well as on test data 

2. **Resolution Enhancement**
   - Images predicted as fake are passed through **ESRGAN** (Enhanced Super-Resolution Generative Adversarial Network) to enhance resolution and reveal finer details.
   - Used in training for half of the dataset , help model classify the artifacts which generalised well to images as low as 32*32 resolution as well.

3. **Artifact Explanation**
   - The enhanced fake images are analyzed using a **pretrained Qwen-2B-Instruct Vision-Language model**, which generates natural language explanations of visual artifacts that led to the "fake" classification.
   - A more specific respone for the images were achieved with the help of prompt engineering and a bit of finetuning for structuring the output formats.

---
## 🧪 Example Workflow

1. Input Image:
   ![input](10_fake.jpg)

2. Output:
   - **Prediction:** Fake
   - **Enhanced Image:** ![enhanced](inference1.jpg)
   - **Explanation:**  An explanation of the present artifacts in the given image which inferred that the image is indeed fake
   

---


## 🛠️ Tools Used

- **TinyViT** – Lightweight Vision Transformer for image classification.
- **ESRGAN** – Deep learning model for image super-resolution.
- **Qwen-VL** – Vision-Language model for generating human-readable explanations.
- **PyTorch** – Deep learning framework.


---



