import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import faiss
import numpy as np
import pandas as pd
import os

class crossModalRetriever:
    """
    Cross-modal retrieval system using CLIP to find visually similar medical cases.
    """
    def __init__(self, model_name_or_path="openai/clip-vit-base-patch32", index_path=None, metadata_path=None, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading CLIP model from {model_name_or_path} to {self.device}...")
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path)
        self.model = CLIPModel.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()
        
        self.index = None
        self.metadata = None
        
        if index_path and os.path.exists(index_path) and metadata_path and os.path.exists(metadata_path):
            self.load_index(index_path, metadata_path)
            
    def load_index(self, index_path, metadata_path):
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        self.metadata = pd.read_csv(metadata_path)
        
    def embed_image(self, image: Image.Image) -> np.ndarray:
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

    def retrieve_similar(self, image: Image.Image, k=3):
        """
        Retrieves the top-k similar images based on CLIP embeddings.
        """
        if self.index is None or self.metadata is None:
            raise ValueError("FAISS index and metadata must be loaded first.")
            
        query_embedding = self.embed_image(image)
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                row = self.metadata.iloc[idx]
                results.append({
                    "image_id": row["image_id"],
                    "caption": row.get("caption", "No caption available"),
                    "score": float(distances[0][i])
                })
                
        return results
