import os
import sys
import json
import requests

def print_banner():
    print("\n" + "=" * 65)
    print("                 DORAEMON CERTIFICATE SCANNER")
    print("  Offline Multi-Layer Visual Extraction Engine by Sachin Ray")
    print("=" * 65)

def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m.get("name") for m in r.json().get("models", [])]
            return True, models
    except Exception:
        pass
    return False, []

def get_file_path():
    while True:
        path = input("\nEnter certificate file path (or drag & drop file here): ").strip().strip('"').strip("'")
        if not path:
            return None
        if os.path.exists(path):
            return path
        print(f"File not found: '{path}'. Please try again.")

def run_multilayer():
    file_path = get_file_path()
    if not file_path:
        return
    
    from src.vlm_multi_layer_extractor import MultiLayerCertificateExtractor
    print("\nRunning Doraemon Multi-Layer Extraction Engine...")
    print("(Parallel Neural OCR + Local Vision-Language Model with Consensus)")
    
    extractor = MultiLayerCertificateExtractor()
    qr_crop = os.path.join("output", "qr_scan.png")
    os.makedirs("output", exist_ok=True)
    
    result = extractor.extract(file_path, output_qr_crop_path=qr_crop)
    
    out_json = os.path.join("output", "extracted_result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        
    print("\n" + "=" * 65)
    print("EXTRACTED CERTIFICATE FIELDS:")
    print("=" * 65)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 65)
    print(f"Results saved to: {out_json}")
    if result.get("qrImage"):
        print(f"QR code cropped : {result['qrImage']}")

def run_classical():
    file_path = get_file_path()
    if not file_path:
        return
        
    from src.certificate_extractor import CertificateExtractor
    print("\nRunning Doraemon Fast Classical OCR & QR Engine...")
    
    extractor = CertificateExtractor()
    qr_crop = os.path.join("output", "qr_scan.png")
    os.makedirs("output", exist_ok=True)
    
    result = extractor.extract(file_path, output_qr_crop_path=qr_crop)
    
    out_json = os.path.join("output", "extracted_result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        
    print("\n" + "=" * 65)
    print("EXTRACTED CERTIFICATE FIELDS:")
    print("=" * 65)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 65)
    print(f"Results saved to: {out_json}")
    if result.get("qrImage"):
        print(f"QR code cropped : {result['qrImage']}")

def run_api_server():
    print("\nStarting local Doraemon FastAPI server on http://localhost:8000 ...")
    print("Press Ctrl+C in terminal to stop the server.\n")
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)

def run_benchmark_suite():
    import run_benchmark
    print("\nRunning Doraemon Evaluation Benchmark Suite...")
    run_benchmark.main()

def main():
    while True:
        print_banner()
        ollama_ok, models = check_ollama()
        
        print("\nSystem Status:")
        print("  - Neural OCR (EasyOCR / OpenCV): Ready")
        if ollama_ok:
            has_qwen = any("qwen2.5vl" in m for m in models)
            print(f"  - Local Ollama Service: Online (qwen2.5vl: {'Installed' if has_qwen else 'Not Found'})")
        else:
            print("  - Local Ollama Service: Offline or not running")

        print("\nSelect an option:")
        print("  [1] Multi-Layer AI Vision + Neural OCR (Highest Accuracy)")
        print("  [2] Fast Classical OCR + QR Engine (Instant Scan)")
        print("  [3] Start Local Web Server (FastAPI on port 8000)")
        print("  [4] Run Evaluation Benchmark")
        print("  [5] Exit")
        
        choice = input("\nEnter choice (1-5) [default: 1]: ").strip()
        if choice in ["", "1"]:
            run_multilayer()
        elif choice == "2":
            run_classical()
        elif choice == "3":
            run_api_server()
        elif choice == "4":
            run_benchmark_suite()
        elif choice == "5":
            print("\nExiting Doraemon. Have a great day!\n")
            break
        else:
            print("Invalid choice, please select 1-5.")

        again = input("\nDo you want to scan another certificate or return to menu? (Y/n): ").strip().lower()
        if again in ["n", "no"]:
            print("\nExiting Doraemon. Goodbye!\n")
            break

if __name__ == "__main__":
    main()
