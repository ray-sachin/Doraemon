import os
import json
import argparse
from typing import Dict, Any
from rapidfuzz import fuzz
from src.certificate_extractor import CertificateExtractor

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    import re
    # Remove extra spaces and special punctuation
    cleaned = re.sub(r'[\r\n\t]+', ' ', str(text))
    cleaned = re.sub(r'\s+', ' ', cleaned).strip().lower()
    return cleaned

def match_field(pred: Any, gt: Any, threshold: float = 75.0) -> bool:
    """
    Compares predicted field value against ground truth value.
    Supports exact match, fuzzy similarity, and substring inclusion.
    """
    if gt is None:
        # If ground truth is None, prediction should ideally be None or empty
        return (pred is None or str(pred).strip() == "")

    if pred is None:
        return False

    norm_pred = normalize_text(pred)
    norm_gt = normalize_text(gt)

    if norm_pred == norm_gt:
        return True

    # Check substring inclusion
    if norm_gt in norm_pred or norm_pred in norm_gt:
        return True

    # Check fuzzy token set ratio
    similarity = fuzz.token_set_ratio(norm_pred, norm_gt)
    return similarity >= threshold

def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Run Doraemon Certificate Benchmark Suite")
    parser.add_argument(
        "--catalog", 
        type=str, 
        default="data/certificates_benchmark/ground_truth.json",
        help="Path to ground truth catalog"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=50, 
        help="Max number of test certificates to evaluate"
    )
    parser.add_argument(
        "--report-out", 
        type=str, 
        default="output/benchmark_report.json",
        help="Destination path for benchmark evaluation report"
    )

    args = parser.parse_args()

    if not os.path.exists(args.catalog):
        print(f"Catalog file not found: {args.catalog}")
        print("Please run `python -m src.dataset_manager` first to generate the benchmark dataset.")
        return

    with open(args.catalog, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    test_items = catalog[:args.limit]
    print("=" * 70)
    print("CERTIFICATE SOTA OCR & EXTRACTION BENCHMARK EVALUATOR")
    print(f"Total Test Certificates: {len(test_items)}")
    print("=" * 70)

    extractor = CertificateExtractor(use_gpu=False)

    field_keys = [
        "recipientName",
        "certificateID",
        "issuerName",
        "institutionName",
        "certificateTitle",
        "courseName",
        "certificateDescription",
        "completionDate",
        "qrImage"
    ]

    field_scores = {k: {"correct": 0, "total": 0} for k in field_keys}
    detailed_results = []

    for idx, item in enumerate(test_items, 1):
        cert_id = item["id"]
        file_path = item["file_path"]
        gt = item["ground_truth"]

        print(f"\n[{idx}/{len(test_items)}] Evaluating: {cert_id} ({os.path.basename(file_path)})")
        
        qr_crop_path = f"output/benchmark_crops/{cert_id}_qr.png"
        try:
            preds = extractor.extract(file_path, output_qr_crop_path=qr_crop_path)
        except Exception as e:
            print(f"  [ERROR] Extraction failed on {file_path}: {e}")
            preds = {k: None for k in field_keys}

        item_result = {
            "id": cert_id,
            "file": file_path,
            "field_matches": {}
        }

        # Check each field
        for field in field_keys:
            field_scores[field]["total"] += 1
            if field == "qrImage":
                # Check if QR code presence matched
                gt_has_qr = gt.get("has_qr", True)
                pred_has_qr = bool(preds.get("qrImage"))
                is_correct = (gt_has_qr == pred_has_qr)
            else:
                gt_val = gt.get(field)
                pred_val = preds.get(field)
                is_correct = match_field(pred_val, gt_val)

            if is_correct:
                field_scores[field]["correct"] += 1

            item_result["field_matches"][field] = {
                "correct": is_correct,
                "predicted": preds.get(field),
                "ground_truth": gt.get(field)
            }
            status = "[PASS]" if is_correct else "[FAIL]"
            print(f"  {field.ljust(22)}: {status} | Pred: {str(preds.get(field))[:35]}")

        detailed_results.append(item_result)

    # Calculate overall summary
    summary_table = {}
    total_correct = 0
    total_evaluated = 0

    print("\n" + "=" * 70)
    print("BENCHMARK SCORECARD ACROSS ALL TEST CERTIFICATES:")
    print("=" * 70)
    print(f"{'Field Name'.ljust(26)} | {'Correct / Total'.ljust(18)} | Accuracy")
    print("-" * 70)

    for field in field_keys:
        correct = field_scores[field]["correct"]
        total = field_scores[field]["total"]
        acc = (correct / total * 100.0) if total > 0 else 0.0
        summary_table[field] = {
            "correct": correct,
            "total": total,
            "accuracy_pct": round(acc, 2)
        }
        total_correct += correct
        total_evaluated += total
        print(f"{field.ljust(26)} | {f'{correct} / {total}'.ljust(18)} | {acc:.1f}%")

    overall_acc = (total_correct / total_evaluated * 100.0) if total_evaluated > 0 else 0.0
    print("=" * 70)
    print(f"OVERALL CERTIFICATE EXTRACTION ACCURACY: {overall_acc:.2f}%")
    print(f"TARGET SLA THRESHOLD                   : >= 90.00%")
    status_label = "PASSED (>= 90%)" if overall_acc >= 90.0 else "NEEDS REFINEMENT (< 90%)"
    print(f"BENCHMARK STATUS                       : {status_label}")
    print("=" * 70)

    report = {
        "overall_accuracy_pct": round(overall_acc, 2),
        "target_threshold_pct": 90.0,
        "status": status_label,
        "field_summary": summary_table,
        "certificates_evaluated": len(test_items),
        "detailed_results": detailed_results
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.report_out)), exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed benchmark report written to: {args.report_out}")

if __name__ == "__main__":
    main()
