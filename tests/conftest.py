import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io

from src.api.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def dummy_image():
    # Create a simple valid image to be used in all tests
    img = Image.new('RGB', (100, 100), color='white')
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return buffered.getvalue()

@pytest.fixture
def image_file(dummy_image):
    return ("test.png", dummy_image, "image/png")
