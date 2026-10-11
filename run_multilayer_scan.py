import os
import sys
import json
import argparse
from src.vlm_multi_layer_extractor import MultiLayerCertificateExtractor

def main():
    parser = argparse.ArgumentParser(description="Doraemon Multi-Layer Certificate Extractor (VLM + OCR)")
    parser.add_argument(
        "--input", 
        type=str, 
        default=r"c:\Users\rayne\Downloads\Diploma in Data Science.pdf",
        help="Path to certificate PDF or image file"
    )
    parser.add_argument(
        "--qr-out", 
        type=str, 
        default=r"output\qr_multilayer.png",
        help="Destination path to save cropped QR code image"
    )
    parser.add_argument(
        "--json-out", 
        type=str, 
        default=r"output\extracted_multilayer.json",
        help="Destination path to save extracted JSON output"
    )
    parser.add_argument(
        "--dim", 
        type=int, 
        default=1024,
        help="Maximum dimension for VLM token budget"
    )
    parser.add_argument(
        "--use-secondary", 
        action="store_true", 
        help="Enable secondary VLM (MiniCPM-V) if downloaded"
    )
    parser.add_argument(
        "--force-vlm", 
        action="store_true", 
        help="Force deep VLM inference even if digital text stream is available"
    )

    args = parser.parse_args()

    print("=" * 65)
    print("DORAEMON MULTI-LAYER CERTIFICATE EXTRACTION ENGINE")
    print(f"Input file       : {args.input}")
    print(f"QR crop out      : {args.qr_out}")
    print(f"VLM Token Dim    : {args.dim}px")
    print(f"Secondary VLM    : {'Enabled (MiniCPM-V)' if args.use_secondary else 'Disabled'}")
    print(f"Force VLM        : {args.force_vlm}")
    print("=" * 65)

    extractor = MultiLayerCertificateExtractor(max_vlm_dim=args.dim)
    result = extractor.extract(
        file_path=args.input,
        output_qr_crop_path=args.qr_out,
        use_secondary_vlm=args.use_secondary,
        force_vlm=args.force_vlm
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("CONSENSUS ARBITRATED CERTIFICATE FIELDS:")
    print("=" * 65)
    print(json.dumps(result, indent=2))
    print("=" * 65)
    print(f"Saved results to: {args.json_out}")
    if result.get("qrImage"):
        print(f"QR code cropped : {result['qrImage']}")
    else:
        print("QR code         : None detected on this certificate")

if __name__ == "__main__":
    main()
