import cv2
import numpy as np
from PIL import Image
import os

def load_and_preprocess_dicom(dicom_path, target_size=(224, 224)):
    """
    Loads a DICOM file, converts it to 8-bit PNG, and returns a PIL Image.
    Requires pydicom.
    """
    try:
        import pydicom
        dcm = pydicom.dcmread(dicom_path)
        pixel_array = dcm.pixel_array
        
        # Normalize to 0-255
        pixel_array = pixel_array - np.min(pixel_array)
        pixel_array = pixel_array / np.max(pixel_array)
        pixel_array = (pixel_array * 255).astype(np.uint8)
        
        # Convert to PIL RGB
        image = Image.fromarray(pixel_array)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        return image.resize(target_size)
    except ImportError:
        print("Warning: pydicom not installed. Cannot process DICOM files.")
        return None
