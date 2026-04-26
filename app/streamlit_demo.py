import streamlit as st
import requests
from PIL import Image
import io
import base64

# Configure Page
st.set_page_config(
    page_title="MedVision-LM Assistant",
    page_icon="⚕️",
    layout="wide"
)

API_URL = "http://localhost:8000/api/v1"

st.title("⚕️ MedVision-LM: Multimodal Medical AI Assistant")
st.markdown("""
Upload a medical image (e.g., Chest X-Ray) to automatically generate clinical captions, 
ask specific questions (VQA), and retrieve visually similar cases from the database.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Image Input")
    uploaded_file = st.file_uploader("Upload Medical Scan...", type=["jpg", "jpeg", "png", "dicom"])
    
    question = st.text_input("Ask a clinical question (optional):", placeholder="e.g., Is there consolidation?")
    
    analyze_btn = st.button("Analyze Image", type="primary", use_container_width=True)

with col2:
    st.subheader("AI Analysis")
    if uploaded_file and analyze_btn:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("Analyzing image across multimodal pipelines..."):
            try:
                # Prepare payload
                files = {"file": ("image.png", uploaded_file.getvalue(), "image/png")}
                data = {"top_k": 3}
                if question:
                    data["question"] = question
                    
                response = requests.post(f"{API_URL}/analyze", files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("Analysis Complete!")
                    
                    # 1. Caption
                    st.markdown("### 📝 Auto-Generated Report")
                    st.info(result.get("caption"))
                    
                    # 2. VQA
                    if result.get("vqa_answer"):
                        st.markdown("### ❓ VQA Response")
                        st.write(f"**Q:** {question}")
                        st.success(f"**A:** {result['vqa_answer']}")
                        
                    # 3. GradCAM
                    if result.get("heatmap_base64"):
                        st.markdown("### 🔥 Attention Heatmap")
                        img_bytes = base64.b64decode(result["heatmap_base64"])
                        heatmap = Image.open(io.BytesIO(img_bytes))
                        st.image(heatmap, caption="GradCAM Attention", use_column_width=True)
                        
                    # 4. Retrieval
                    st.markdown("### 🔍 Similar Reference Cases")
                    matches = result.get("retrieval_matches", [])
                    
                    if not matches:
                        st.warning("No similar cases found or FAISS index not loaded.")
                    else:
                        cols = st.columns(len(matches))
                        for i, match in enumerate(matches):
                            with cols[i]:
                                st.metric(label="Similarity Score", value=f"{match['score']:.4f}")
                                st.caption(f"ID: {match['image_id']}")
                                st.write(match['caption'])
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                st.error(f"Failed to connect to API. Is it running? Error: {e}")
    elif not uploaded_file and analyze_btn:
        st.warning("Please upload an image first.")
