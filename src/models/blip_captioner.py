import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

class MedicalImageCaptioner:
    """
    Wrapper for BLIP Image Captioning fine-tuned on medical datasets.
    """
    def __init__(self, model_name_or_path="Salesforce/blip-image-captioning-base", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading BLIP Captioning model from {model_name_or_path} to {self.device}...")
        self.processor = BlipProcessor.from_pretrained(model_name_or_path)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()

    def generate_caption(self, image: Image.Image, max_length=50) -> str:
        """
        Generates a descriptive caption for the given medical image.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=max_length)
            
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
