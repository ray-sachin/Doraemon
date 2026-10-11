import os
import sys
import json
import argparse
from src.certificate_extractor import CertificateExtractor

def main():
    parser = argparse.ArgumentParser(description="Extract fields from certificate PDF or Image.")
    parser.add_argument(
        "--input", 
        type=str, 
        default=None,
        help="Path to certificate PDF or image file (prompts if omitted)"
    )
    parser.add_argument(
        "--qr-out", 
        type=str, 
        default=r"output\qr_extracted.png",
        help="Destination path to save cropped QR code image"
    )
    parser.add_argument(
        "--json-out", 
        type=str, 
        default=r"output\extracted_fields.json",
        help="Destination path to save extracted JSON output"
    )
    parser.add_argument(
        "--gpu", 
        action="store_true", 
        help="Enable GPU acceleration if available"
    )

    args = parser.parse_args()

    input_path = args.input
    if not input_path:
        input_path = input("Enter certificate file path (or drag & drop file here): ").strip().strip('"').strip("'")

    if not input_path or not os.path.exists(input_path):
        print(f"Error: Certificate file not found at '{input_path}'")
        return

    print("=" * 60)
    print("DORAEMON CERTIFICATE EXTRACTION ENGINE")
    print(f"Input file : {input_path}")
    print(f"QR crop out: {args.qr_out}")
    print("=" * 60)

    extractor = CertificateExtractor(use_gpu=args.gpu)
    result = extractor.extract(input_path, output_qr_crop_path=args.qr_out)

    # Save output JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("EXTRACTED CERTIFICATE FIELDS:")
    print("=" * 60)
    print(json.dumps(result, indent=2))
    print("=" * 60)
    print(f"Results successfully saved to: {args.json_out}")
    if result.get("qrImage"):
        print(f"QR code image saved to       : {result['qrImage']}")
    else:
        print("QR code                      : None detected on this certificate")

if __name__ == "__main__":
    main()
