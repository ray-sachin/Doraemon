# Doraemon

> An offline, multi-layer certificate and credential extraction engine combining computer vision, neural OCR, and local vision-language models.

Developed by **Sachin Kumar Ray**.

---

## Overview

Extracting structured data from certificates is surprisingly tricky in the real world:
- Some files are born-digital vector PDFs.
- Some are flat scans or mobile photos taken under harsh shadows, uneven lighting, or perspective tilt.
- Many have handwritten student roll numbers or cursive calligraphy that confuse standard OCR models.
- Uploading academic credentials or sensitive institutional records to third-party cloud APIs often violates privacy guidelines and introduces latency.

**Doraemon** solves this by running **100% offline on local hardware**. It uses a multi-tier pipeline that routes clean digital PDFs through a fast programmatic path, while running camera photos and scanned documents through computer vision preprocessing, neural OCR, and local Vision-Language Models (Qwen2.5-VL and MiniCPM-V via Ollama). A consensus arbitration layer reconciles the outputs to eliminate hallucinations and output clean, verified JSON.

---

## How It Works

```
                        Input Certificate (PDF or Image)
                                       │
                        ┌──────────────┴──────────────┐
                        ▼                             ▼
               [Digital Vector PDF?]          [Scanned / Camera Photo]
                        │                             │
              YES ──────┼────── NO                    │
              │                                       ▼
              ▼                          [CV Preprocessing]
       [Fast-Path Engine]                - Hough Transform Deskewing
       - Vector text parsing             - Morphological Background Division
       - Regex layout anchors            - CLAHE Contrast Normalization
       - Execution: < 50ms               - Padded QR Detection & Crop
              │                                       │
              │                               ┌───────┴───────┐
              │                               ▼               ▼
              │                         [Stream A]       [Stream B]
              │                         Neural OCR        Qwen2.5-VL (3B)
              │                         (EasyOCR)         via local Ollama
              │                               │               │
              │                               └───────┬───────┘
              │                                       │
              │                                       ▼
              │                           [Initial Arbitration]
              │                                       │
              │                         Confidence < 0.85 or Ungrounded?
              │                                       │
              │                               YES ────┴──── NO
              │                               │              │
              │                               ▼              │
              │                          [Stream C]          │
              │                         MiniCPM-V            │
              │                       (Auto Escalation)      │
              │                               │              │
              │                               └───────┬──────┘
              │                                       │
              ▼                                       ▼
     ┌────────────────────────────────────────────────────────┐
     │          Consensus & Stroke Arbitration Layer          │
     │  - Boilerplate prefix stripping (Mr./Ms., Mr./Mrs.)   │
     │  - OCR character token grounding                      │
     │  - Handwritten stroke disambiguation (e.g. BT24 vs RT)│
     │  - Clean 9-field output JSON formatting               │
     └────────────────────────────────────────────────────────┘
```

---

## Key Features

1. **Fast-Path Digital Stream (< 50ms)**:
   Extracts born-digital vector text directly using `pypdfium2` without invoking neural models when text quality is already high.

2. **Computer Vision Restoration**:
   - **Hough Line Deskewing**: Straightens tilted photos taken with mobile cameras (up to ±25 degrees).
   - **Morphological Background Division**: Normalizes uneven lighting and removes harsh camera shadows.
   - **CLAHE Contrast Normalization**: Enhances faint or degraded ink in LAB color space.
   - **OpenCV QR Code Cropping**: Detects, decodes, and saves QR codes with tight padding.

3. **Multi-Model Parallel Inference**:
   - Dispatches neural OCR (EasyOCR CRAFT + CRNN) and VLM inference (Qwen2.5-VL:3B) concurrently using Python's `ThreadPoolExecutor`.
   - Resolution budgeting caps visual dimension to 1024px, cutting visual patch tokens and speeding up inference significantly.

4. **Autonomous Low-Confidence Escalation**:
   - If any core entity confidence drops below 0.85 or is flagged as ungrounded, the engine automatically engages `minicpm-v` as a secondary tie-breaker arbitrator.

5. **Cross-Stream Stroke Disambiguation**:
   - In cursive or slanted handwriting, optical character ambiguities are resolved by cross-checking neural OCR character tokens against document layout context.
   - Pre-printed form boilerplate (`Mr./Ms.`, `Mr./Mrs.`) is stripped so only the clean recipient name is retained.

6. **100% Offline & Private**:
   - No cloud API keys, external servers, or data transmission. Everything runs locally on your CPU/GPU.

---

## Output Schema

The output strictly produces the following 9 target fields:

```json
{
  "recipientName": "Sachin Ray",
  "certificateID": null,
  "certificateTitle": "CERTIFICATE",
  "courseName": "Adventure and Community Service's Club",
  "institutionName": "National Institute of Technology Uttarakhand",
  "issuerName": "Dr. Kuldeep Singh",
  "certificateDescription": "for his hard work and active participation in the Adventure and Community Service's Club as Associate Member for the session 2024-25.",
  "completionDate": null,
  "qrImage": null,
  "_metadata": {
    "total_latency_seconds": 80.941,
    "stream_ocr_duration": 10.85,
    "stream_vlm_duration": 28.244,
    "token_budget_dim": 1024,
    "models_engaged": [
      "qwen2.5vl:3b",
      "minicpm-v"
    ],
    "audit": {
      "confidences": {
        "recipientName": 0.98,
        "certificateID": 0.0,
        "certificateTitle": 0.92,
        "courseName": 0.92,
        "institutionName": 0.99,
        "issuerName": 0.99,
        "certificateDescription": 0.99
      }
    }
  }
}
```

---

---

## Quickstart (One-Click)

If you are on Windows, you do not need to manually configure Python environments or download packages:
1. Double-click **`start.bat`**.
2. The launcher will automatically:
   - Check if Python is installed (and offer to install Python 3.11 if missing).
   - Create an isolated virtual environment (`.venv`).
   - Prompt you to install required dependencies (`requirements.txt`).
   - Check if Ollama and both local vision models (`qwen2.5vl:3b` and `minicpm-v:latest`) are present (and offer to download them).
   - Launch the scanner where you can simply **paste the path or drag and drop** your certificate to extract fields immediately!

---

## Manual Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/ray-sachin/Doraemon.git
cd Doraemon
```

### 2. Install Python Dependencies
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Install and Configure Ollama (for VLM inference)
Install [Ollama](https://ollama.com/) and pull the local vision models:
```bash
ollama pull qwen2.5vl:3b
ollama pull minicpm-v:latest
```

---

## Usage

### 1. Run Multi-Layer Scan (VLM + OCR)
The multi-layer pipeline runs digital triage, parallel OCR + VLM, and consensus arbitration:
```bash
# Process a certificate
python run_multilayer_scan.py --input "path/to/certificate.pdf"

# Force deep VLM extraction even on born-digital PDFs
python run_multilayer_scan.py --input "path/to/certificate.pdf" --force-vlm

# Enable dual-VLM consensus manually
python run_multilayer_scan.py --input "path/to/certificate.pdf" --use-secondary
```

### 2. Run High-Speed Classical OCR Scan
For rapid batch processing using computer vision restoration and EasyOCR:
```bash
python run_certificate_scan.py --input "path/to/certificate.pdf"
```

### 3. Run Benchmark Suite
Evaluate extraction accuracy on the test catalog:
```bash
python run_benchmark.py
```

### 4. Run Local FastAPI Server
Launch a local REST API endpoint for integration:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
API endpoint: `POST http://localhost:8000/extract-certificate`

---

## Project Structure

```
Doraemon/
|-- data/
|   `-- certificates_benchmark/   # Benchmark test certificates and ground truth catalog
|-- src/
|   |-- api.py                    # FastAPI server
|   |-- certificate_extractor.py  # Classical CV + OCR extraction engine
|   |-- dataset_manager.py        # Dataset generation & catalog manager
|   `-- vlm_multi_layer_extractor.py # Multi-layer VLM + OCR extraction engine
|-- run_benchmark.py              # Benchmark evaluation script
|-- run_certificate_scan.py       # High-speed OCR CLI runner
|-- run_multilayer_scan.py        # Multi-layer VLM + OCR CLI runner
|-- requirements.txt              # Core project dependencies
`-- README.md                     # Documentation
```

---

## License

MIT License. Free for academic, personal, and commercial use.
