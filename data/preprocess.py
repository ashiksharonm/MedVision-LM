import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image

def process_image(image_path, target_size=(224, 224)):
    """
    Standardize images:
    - Resize to target size.
    - Convert to RGB.
    - Normalize pixel intensities (optional).
    """
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    return img

def build_datasets(data_dir="data", output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    metadata_frames = []
    
    # Process IU X-Ray if exists
    iu_path = os.path.join(data_dir, "iuxray", "metadata.csv")
    if os.path.exists(iu_path):
        df_iu = pd.read_csv(iu_path)
        metadata_frames.append(df_iu)
        print(f"Found {len(df_iu)} IU X-Ray entries.")

    # Process ROCO if exists
    roco_path = os.path.join(data_dir, "roco", "metadata.csv")
    if os.path.exists(roco_path):
        df_roco = pd.read_csv(roco_path)
        metadata_frames.append(df_roco)
        print(f"Found {len(df_roco)} ROCO entries.")

    if not metadata_frames:
        print("No metadata found. Please run download scripts first.")
        return

    combined_df = pd.concat(metadata_frames, ignore_index=True)
    processed_records = []
    
    print("Processing combined dataset...")
    for idx, row in combined_df.iterrows():
        try:
            img_path = row["image_path"]
            if not os.path.exists(img_path):
                continue
                
            processed_img = process_image(img_path)
            
            # Save processed image
            new_filename = f"proc_{idx}_{os.path.basename(img_path)}"
            new_path = os.path.join(output_dir, "images", new_filename)
            processed_img.save(new_path)
            
            processed_records.append({
                "image_id": new_filename,
                "image_path": new_path,
                "caption": row["caption"]
            })
            
            if idx % 100 == 0 and idx > 0:
                print(f"Processed {idx} images.")
        except Exception as e:
            print(f"Error on {row['image_path']}: {e}")
            
    final_df = pd.DataFrame(processed_records)
    
    # Create splits
    train_df = final_df.sample(frac=0.8, random_state=42)
    val_test_df = final_df.drop(train_df.index)
    val_df = val_test_df.sample(frac=0.5, random_state=42)
    test_df = val_test_df.drop(val_df.index)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"Processing complete. Splits saved to {output_dir}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

if __name__ == "__main__":
    build_datasets()
