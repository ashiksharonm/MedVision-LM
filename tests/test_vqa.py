def test_vqa_endpoint(client, image_file, monkeypatch):
    class MockVQA:
        def answer_question(self, image, question):
            return f"Answer for: {question}"
            
    monkeypatch.setattr("src.api.routes.vqa.get_vqa", lambda: MockVQA())
    
    question = "Is there consolidation?"
    response = client.post("/api/v1/vqa", data={"question": question}, files={"file": image_file})
    
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["question"] == question
    assert res_json["answer"] == f"Answer for: {question}"

def test_vqa_missing_question(client, image_file):
    # Form data missing `question`
    response = client.post("/api/v1/vqa", files={"file": image_file})
    assert response.status_code == 422 # FastAPI validation error
