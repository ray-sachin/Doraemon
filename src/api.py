import os
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.certificate_extractor import CertificateExtractor

app = FastAPI(
    title="Doraemon Certificate Extraction API",
    description="Offline information extraction for certificates (PDF & Images)",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize extractor instance
extractor = CertificateExtractor(use_gpu=False)

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Doraemon Extraction API",
        "supported_formats": ["pdf", "png", "jpg", "jpeg", "webp", "tiff"]
    }

@app.post("/extract-certificate")
async def extract_certificate(
    file: UploadFile = File(...),
    return_base64_qr: bool = Query(True, description="Return QR image as base64 data URL in JSON")
):
    """
    Upload certificate file (PDF or Image) and retrieve structured fields:
    - recipientName
    - certificateID
    - issuerName
    - institutionName
    - certificateTitle
    - courseName
    - certificateDescription
    - completionDate
    - qrImage
    """
    filename = file.filename or "upload_temp"
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_file_path = os.path.join(tmp_dir, filename)
        qr_crop_path = os.path.join(tmp_dir, "qr_extracted.png")

        with open(input_file_path, "wb") as f_out:
            shutil.copyfileobj(file.file, f_out)

        try:
            fields = extractor.extract(input_file_path, output_qr_crop_path=qr_crop_path)
            
            # If user requested base64 instead of temp disk path
            if return_base64_qr and os.path.exists(qr_crop_path):
                import base64
                with open(qr_crop_path, "rb") as qr_f:
                    b64_str = base64.b64encode(qr_f.read()).decode("utf-8")
                    fields["qrImage"] = f"data:image/png;base64,{b64_str}"

            return JSONResponse(content={
                "success": True,
                "data": fields
            })

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
