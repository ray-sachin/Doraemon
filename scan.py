import os
import sys
import json
import requests

def print_header(ollama_online: bool, model_ready: bool):
    print("\n" + "=" * 65)
    print("                 DORAEMON CERTIFICATE SCANNER")
    print("  Offline Multi-Layer Visual Extraction Engine by Sachin Ray")
    print("=" * 65)
    print("System Status:")
    print("  - Neural OCR (EasyOCR / OpenCV)  : Ready")
    if ollama_online and model_ready:
        print("  - Local AI Vision (Qwen2.5-VL)   : Ready (Ollama Online)")
    elif ollama_online:
        print("  - Local AI Vision (Qwen2.5-VL)   : Model not pulled (Running Classical OCR)")
    else:
        print("  - Local AI Vision (Ollama)       : Offline (Running Classical OCR)")
    print("=" * 65)

def check_ollama_status():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            models = [m.get("name", "").lower() for m in r.json().get("models", [])]
            has_qwen = any("qwen2.5vl" in m for m in models)
            return True, has_qwen
    except Exception:
        pass
    return False, False

def clean_input_path(raw: str) -> str:
    if not raw:
        return ""
    # Strip leading/trailing whitespaces, double inverted commas, single quotes
    cleaned = raw.strip()
    # Handle PowerShell copy-paste prefix: & "C:\..."
    if cleaned.startswith("&"):
        cleaned = cleaned[1:].strip()
    # Strip surrounding quotes repeatedly in case of nested quotes
    while (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()
    # Expand user home if ~ is used
    cleaned = os.path.expanduser(cleaned)
    return cleaned

def prompt_for_file():
    while True:
        raw_input = input("\nPaste your certificate file location (or drag & drop file here): ")
        
        # User wants to exit
        if not raw_input.strip() or raw_input.strip().lower() in ["q", "quit", "exit"]:
            return None

        file_path = clean_input_path(raw_input)
        
        # Validate existence
        if os.path.isfile(file_path):
            return file_path
        
        # Friendly troubleshooting guidance
        print("\n" + "-" * 65)
        print(f"[!] Error: File not found at:\n    {file_path}")
        print("\nIt looks like the path is incorrect or incomplete.")
        print("Tips:")
        print("  1. In Windows File Explorer, right-click your file and choose 'Copy as path'.")
        print("  2. Or simply drag and drop the certificate file directly into this window.")
        print("  3. Make sure the file exists and is not open in another locking application.")
        print("-" * 65)

def scan_certificate(file_path: str, use_vlm: bool):
    os.makedirs("output", exist_ok=True)
    qr_crop_path = os.path.join("output", "qr_scan.png")
    out_json_path = os.path.join("output", "extracted_result.json")
    
    print(f"\nProcessing: {os.path.basename(file_path)}")
    print("Scanning and extracting certificate fields... please wait a moment.\n")

    if use_vlm:
        from src.vlm_multi_layer_extractor import MultiLayerCertificateExtractor
        extractor = MultiLayerCertificateExtractor()
        result = extractor.extract(file_path, output_qr_crop_path=qr_crop_path)
    else:
        from src.certificate_extractor import CertificateExtractor
        extractor = CertificateExtractor()
        result = extractor.extract(file_path, output_qr_crop_path=qr_crop_path)

    # Save output JSON
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Pretty print final fields
    print("=" * 65)
    print("EXTRACTED CERTIFICATE FIELDS:")
    print("=" * 65)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 65)
    print(f"Results saved to : {out_json_path}")
    if result.get("qrImage"):
        print(f"QR code cropped  : {result['qrImage']}")
    else:
        print("QR code          : None detected on this certificate")

def main():
    ollama_online, qwen_ready = check_ollama_status()
    use_vlm = ollama_online and qwen_ready

    print_header(ollama_online, qwen_ready)

    while True:
        file_path = prompt_for_file()
        if not file_path:
            print("\nExiting Doraemon. Have a great day!\n")
            break

        try:
            scan_certificate(file_path, use_vlm=use_vlm)
        except Exception as e:
            print(f"\n[!] Extraction error: {e}")
            print("Please verify the file is a readable PDF or image.")

        print("\n" + "=" * 65)
        print("Ready for next certificate.")

if __name__ == "__main__":
    main()
