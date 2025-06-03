# FakeImage-Classifier
This project implements a multi-stage pipeline for analyzing images to detect fakes and understand the artifacts that betray them:

Fake/Real Classification:
A Tiny ViT (Vision Transformer) model trained to classify input images as either real or fake.

Resolution Enhancement:
Images classified as fake are enhanced using ESRGAN (Enhanced Super-Resolution Generative Adversarial Network) to improve image quality and details.

Artifact Explanation:
The enhanced fake images are then processed by a pretrained Qwen Vision-Language model to generate natural language explanations describing the suspicious artifacts or regions in the image.
