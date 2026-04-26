def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_caption_endpoint(client, image_file, monkeypatch):
    # Mock the captioner prediction to avoid downloading large models during unit tests
    class MockCaptioner:
        def generate_caption(self, image):
            return "Mock medical caption"
    
    monkeypatch.setattr("src.api.routes.caption.get_captioner", lambda: MockCaptioner())
    
    response = client.post("/api/v1/caption", files={"file": image_file})
    assert response.status_code == 200
    assert "caption" in response.json()
    assert response.json()["caption"] == "Mock medical caption"

def test_caption_invalid_file(client):
    invalid_file = ("test.txt", b"not an image", "text/plain")
    response = client.post("/api/v1/caption", files={"file": invalid_file})
    assert response.status_code == 400
