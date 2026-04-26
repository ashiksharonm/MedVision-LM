import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image

class MedicalVQA:
    """
    Wrapper for BLIP Visual Question Answering fine-tuned for medical purposes.
    """
    def __init__(self, model_name_or_path="Salesforce/blip-vqa-base", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading BLIP VQA model from {model_name_or_path} to {self.device}...")
        self.processor = BlipProcessor.from_pretrained(model_name_or_path)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()

    def answer_question(self, image: Image.Image, question: str, max_length=50) -> str:
        """
        Answers a specific clinical question based on the image.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=max_length)
            
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        return answer
