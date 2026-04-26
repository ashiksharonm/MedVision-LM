import os
from datasets import load_dataset
import pandas as pd
from PIL import Image

def download_roco(save_dir="data/roco"):
    """
    Downloads the ROCO dataset from HuggingFace Hub and saves it locally.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    
    print("Loading ROCO dataset from HuggingFace...")
    # Using a subset for demo
    dataset = load_dataset("radiology", "roco", split="train[:1%]")
    
    metadata = []
    
    print("Downloading and processing ROCO images...")
    for i, item in enumerate(dataset):
        try:
            # Check keys based on ROCO dataset structure
            if "image" not in item:
                continue
                
            image = item["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            img_filename = f"roco_{i}.png"
            img_path = os.path.join(save_dir, "images", img_filename)
            image.save(img_path)
            
            caption = item.get("caption", "Medical radiology image.")
            
            metadata.append({
                "image_id": img_filename,
                "image_path": img_path,
                "caption": caption
            })
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(save_dir, "metadata.csv"), index=False)
    print(f"Done! {len(df)} ROCO images saved to {save_dir}")

if __name__ == "__main__":
    download_roco()
