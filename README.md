# MedVision-LM: Medical Visual Language Model Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange.svg)](https://huggingface.co/docs/transformers/index)

> **Status: experimental scaffold / work in progress.** This repo wires pretrained Hugging Face models behind a FastAPI service and Streamlit demo. It is **not** a validated clinical tool, no model here has been fine-tuned and evaluated on medical data, and no performance numbers are reported. Do not use it for medical decisions.

## What the code actually does

- **Captioning** (`src/models/blip_captioner.py`): wraps `Salesforce/blip-image-captioning-base`. A fine-tuning script exists (`src/training/train_blip_caption.py`) but has not been run against real radiology reports, so the default output is a generic image caption.
- **VQA** (`src/models/blip_vqa.py`): wraps `Salesforce/blip-vqa-base`; `train_blip_vqa.py` is a template that expects a user-supplied `image_path, question, answer` CSV.
- **Retrieval** (`src/models/clip_retriever.py`): CLIP (`openai/clip-vit-base-patch32`) embeddings with a FAISS index built by `src/training/build_retrieval_index.py`. Retrieval quality has not been measured.
- **Attention heatmap** (`src/models/gradcam.py`): a placeholder. It draws a fixed Gaussian blob at the image center and does not use model gradients or attention, so it carries no diagnostic information.
- **Metrics** (`src/utils/metrics.py`): `compute_cider_bleu` currently returns zeros (not implemented); `compute_recall_at_k` is an unvalidated helper.
- **API / UI**: FastAPI routes for caption, VQA, retrieve and analyze, plus a Streamlit demo.

## Data

- `data/download_iuxray.py` actually downloads a 1% subset of the **NIH Chest X-ray** dataset (`alkzar90/NIH-Chest-X-ray-dataset`) as a stand-in, and the captions it produces are dummy/placeholder text, not radiology reports.
- `data/download_roco.py` is provided but ROCO has not been used for fine-tuning.
- VQA-RAD is not integrated.

## TODO

- [ ] Real IU X-Ray (image + report) ingestion, with patient-level train/val/test splits
- [ ] Fine-tune BLIP captioning on real reports and evaluate with BLEU/CIDEr (implement `compute_cider_bleu`)
- [ ] Build and evaluate the CLIP/FAISS retrieval index (Recall@K on held-out data)
- [ ] Replace the placeholder heatmap with real Grad-CAM/attention on the vision encoder; validate against expert annotations
- [ ] VQA fine-tuning on VQA-RAD and evaluation
- [ ] Measure API latency on stated hardware
- [ ] Clinical-safety review and disclaimers before any real-world use

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/ashiksharonm/MedVision-LM.git
cd MedVision-LM

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
python data/download_iuxray.py
python data/preprocess.py
```

### 3. Run the API & UI
```bash
# Start FastAPI backend
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Start Streamlit UI (in a new terminal)
streamlit run app/streamlit_demo.py
```

---

## Deployment (Docker)

```bash
docker-compose up --build
```
