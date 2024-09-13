from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import torch
import pandas as pd
import os

def hf_clip_predict(model, processor, text_labels, images):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    text = [f"A photo of a {label}" for label in text_labels]
    inputs = processor(text=text, images=images, return_tensors="pt", padding=True).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return probs

def get_images_from_ids(data_dir, article_ids):
    images = []
    image_ids = []
    for article_id in article_ids:
        image_path = f"{data_dir}/images/0{str(article_id)[:2]}/0{article_id}.jpg"
        if os.path.exists(image_path):
            image = Image.open(image_path)
            images.append(image)
            image_ids.append(article_id)
    return images, image_ids

def get_images_recursive(data_dir):
    images = []
    image_ids = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                image = Image.open(image_path)
                images.append(image)
                image_ids.append(file.split(".")[0])
    return images, image_ids