def test_retrieve_endpoint(client, image_file, monkeypatch):
    class MockRetriever:
        def __init__(self):
            self.index = True # Mock that index is loaded
            
        def retrieve_similar(self, image, k):
            return [{"image_id": f"mock_{i}", "caption": "mock cap", "score": 0.99 - (i * 0.01)} for i in range(k)]
            
    monkeypatch.setattr("src.api.routes.retrieve.get_retriever", lambda: MockRetriever())
    
    response = client.post("/api/v1/retrieve", data={"top_k": 2}, files={"file": image_file})
    
    assert response.status_code == 200
    matches = response.json().get("matches", [])
    assert len(matches) == 2
    assert matches[0]["image_id"] == "mock_0"
    
def test_retrieve_endpoint_no_index(client, image_file, monkeypatch):
    class MockRetrieverNoIndex:
        def __init__(self):
            self.index = None
            
    monkeypatch.setattr("src.api.routes.retrieve.get_retriever", lambda: MockRetrieverNoIndex())
    
    response = client.post("/api/v1/retrieve", files={"file": image_file})
    assert response.status_code == 503
    assert "FAISS index not loaded" in response.json()["detail"]
