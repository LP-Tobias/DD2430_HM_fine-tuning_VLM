from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import torch
import pandas as pd
import os
import numpy as np

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
    for article_id in article_ids:
        image_path = f"{data_dir}/images/0{str(article_id)[:2]}/0{article_id}.jpg"
        if os.path.exists(image_path):
            image = np.array(Image.open(image_path))
            yield image, article_id

def get_images_recursive(data_dir):
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                image = np.array(Image.open(image_path))
                yield image, file.split(".")[0]

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, article_ids, processor=None):
        self.data_dir = data_dir
        self.processor = processor
        # self.resize = resize
        self.image_ids = []
        self.image_paths = []

        for article_id in article_ids:
            image_path = f"{data_dir}/images/0{str(article_id)[:2]}/0{article_id}.jpg"
            if os.path.exists(image_path):
                self.image_ids.append(article_id)
                self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt", padding=True)
            image = inputs["pixel_values"][0]
        return image, self.image_ids[idx]
