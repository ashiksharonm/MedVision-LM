import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer
from PIL import Image

class MedicalCaptionDataset(Dataset):
    def __init__(self, csv_file, processor, max_length=50):
        self.data_frame = pd.read_csv(csv_file)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        image_path = row['image_path']
        caption = row['caption']
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # Fallback for broken images during training
            image = Image.new('RGB', (224, 224), color='black')

        encoding = self.processor(images=image, text=caption, padding="max_length", return_tensors="pt")
        # Remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding

def train(data_path="data/processed", output_dir="./checkpoints/blip-caption-medvision"):
    model_id = "Salesforce/blip-image-captioning-base"
    
    print(f"Loading processor and model: {model_id}")
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    
    train_csv = os.path.join(data_path, "train.csv")
    val_csv = os.path.join(data_path, "val.csv")
    
    if not os.path.exists(train_csv):
        print(f"Error: {train_csv} not found. Run preprocessing first.")
        return
        
    print("Preparing datasets...")
    train_dataset = MedicalCaptionDataset(train_csv, processor)
    val_dataset = MedicalCaptionDataset(val_csv, processor)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=50,
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving final model to {output_dir}...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print("Done!")

if __name__ == "__main__":
    train()
