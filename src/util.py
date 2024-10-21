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

def get_image_paths_and_labels_from_df(df, data_dir):
    article_ids = df["article_id"].values
    image_paths = []
    labels = []
    
    for article_id in article_ids:
        image_path = f"{data_dir}/images/0{str(article_id)[:2]}/0{article_id}.jpg"
        # Check if the image file exists
        if os.path.exists(image_path):
            image_paths.append(image_path)
            # Add corresponding label only if the image exists
            labels.append(df[df["article_id"] == article_id])

    return image_paths, labels

def get_image_paths_and_labels_ordered(df, data_dir):
    article_ids = df["article_id"].values
    image_paths = []
    labels = []
    for article_id in article_ids:
        image_path = f"{data_dir}/images/0{str(article_id)[:2]}/0{article_id}.jpg"
        if os.path.exists(image_path):
            image_paths.append(image_path)
            labels.append(df[df["article_id"] == article_id])
    
    return image_paths, labels

def get_image_paths_and_labels(df, data_dir):
    image_paths = []
    labels = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
                article_id = int(file.split(".")[0])
                labels.append(df[df["article_id"] == article_id])

    return image_paths, labels

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, processor=None):
        self.image_paths = image_paths
        self.processor = processor
        self.image_ids = []

        for image_path in self.image_paths:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image {image_path} not found.")
            else:
                image_id = int(image_path.split("/")[-1].split(".")[0])
                self.image_ids.append(image_id)
            

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt", padding=True)
            image = inputs["pixel_values"][0]
        return image, self.image_ids[idx]
