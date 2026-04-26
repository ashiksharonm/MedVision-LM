import os
import torch
import faiss
import numpy as np
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

def build_index(data_csv="data/processed/train.csv", model_id="openai/clip-vit-base-patch32", save_dir="data/artifacts"):
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Loading CLIP model {model_id}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()
    
    print(f"Loading metadata from {data_csv}...")
    df = pd.read_csv(data_csv)
    
    embedding_dim = model.config.projection_dim
    index = faiss.IndexFlatL2(embedding_dim)
    
    embeddings = []
    valid_indices = []
    
    print("Generating embeddings...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            image_path = row['image_path']
            image = Image.open(image_path).convert("RGB")
            
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                features = model.get_image_features(**inputs)
                
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.cpu().numpy()[0])
            valid_indices.append(idx)
        except Exception as e:
            print(f"Failed to process {row['image_path']}: {e}")
            
    if not embeddings:
        print("No embeddings generated.")
        return
        
    embeddings_np = np.stack(embeddings).astype('float32')
    print(f"Adding {len(embeddings_np)} embeddings to FAISS index...")
    index.add(embeddings_np)
    
    index_path = os.path.join(save_dir, "clip_faiss.index")
    faiss.write_index(index, index_path)
    print(f"Index saved to: {index_path}")
    
    # Save aligned metadata
    valid_df = df.iloc[valid_indices].reset_index(drop=True)
    metadata_path = os.path.join(save_dir, "retrieval_metadata.csv")
    valid_df.to_csv(metadata_path, index=False)
    print(f"Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    build_index()
