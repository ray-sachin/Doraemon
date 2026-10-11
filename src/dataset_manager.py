import os
import shutil
import json
import random
import qrcode
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pypdfium2 as pdfium

DATASET_ROOT = os.path.abspath("data/certificates_benchmark")
REAL_CERT_DIR = os.path.join(DATASET_ROOT, "real")
SYNTH_CERT_DIR = os.path.join(DATASET_ROOT, "synthetic")
GROUND_TRUTH_FILE = os.path.join(DATASET_ROOT, "ground_truth.json")

SAMPLE_INSTITUTIONS = [
    "INSTITUTE OF INDIAN CIVIL STRUCTURES",
    "NATIONAL INSTITUTE OF TECHNOLOGY UTTARAKHAND",
    "INDIAN INSTITUTE OF TECHNOLOGY ROORKEE",
    "HARVARD EXTENSION SCHOOL",
    "STANFORD ARTIFICIAL INTELLIGENCE LAB",
    "GLOBAL DATA SCIENCE ACADEMY",
    "MINISTRY OF YOUTH AFFAIRS & SPORTS",
    "ALL INDIA COUNCIL FOR TECHNICAL EDUCATION"
]

SAMPLE_TITLES = [
    "CERTIFICATE OF COMPLETION",
    "CERTIFICATE OF PARTICIPATION",
    "CERTIFICATE OF ACHIEVEMENT",
    "CERTIFICATE OF EXCELLENCE",
    "CERTIFICATE OF APPRECIATION"
]

SAMPLE_RECIPIENTS = [
    "Vinay Kumar",
    "Sachin Ray",
    "Aarav Sharma",
    "Priya Patel",
    "Rohit Verma",
    "Ananya Deshmukh",
    "Michael Chang",
    "Emily Watson"
]

SAMPLE_COURSES = [
    "SOIL ANALYSIS",
    "GEOTECHNICAL FOUNDATION ENGINEERING",
    "NEXTGEN PATHWAYS IN VLSI SYSTEMS",
    "DEEP LEARNING AND COMPUTER VISION",
    "REINFORCEMENT LEARNING WITH PYTORCH",
    "STRUCTURAL SEISMIC DYNAMICS",
    "CLOUD DEVOPS AND KUBERNETES",
    "VIKSIT BHARAT YOUNG LEADERS DIALOGUE"
]

SAMPLE_ISSUERS = [
    ("Ram Lal Yadav", "HOD Civil Engineering Department"),
    ("Dr. Pradumn Kumar Pandey", "Professor & Course Coordinator"),
    ("Dr. Rajesh Sengupta", "Dean of Academic Affairs"),
    ("Prof. Katherine Miller", "Director of Applied Sciences"),
    ("Vikram Singhania", "President & Chief Secretary")
]

SAMPLE_DATES = [
    "October 8, 2026",
    "November 15, 2025",
    "December 20, 2024",
    "January 12, 2026",
    "March 28, 2025",
    "August 14, 2026"
]

SAMPLE_DESCRIPTIONS = [
    "for outstanding dedication and successful completion of the prestigious program",
    "for active participation and successful contribution to the specialized workshop",
    "for demonstrating remarkable excellence and fulfilling all course requirements",
    "in recognition of outstanding academic performance and dedicated project submission"
]

def collect_local_certificates():
    """
    Collects existing local certificates from user's Downloads into benchmark directory.
    """
    os.makedirs(REAL_CERT_DIR, exist_ok=True)
    downloads_dir = r"c:\Users\rayne\Downloads"
    
    known_targets = [
        "certificate_soil.pdf",
        "certificate.png",
        "certificate (1).png",
        "sachin_ray_alp_2024_certificate_of_completion.pdf",
        "Certificate_STC_NextGen_Mr. Sachin Kumar Ray.pdf"
    ]
    
    collected = []
    for fname in known_targets:
        src = os.path.join(downloads_dir, fname)
        if os.path.exists(src):
            dst = os.path.join(REAL_CERT_DIR, fname)
            shutil.copy2(src, dst)
            collected.append(dst)
            print(f"[Collected Real] {fname} -> {dst}")
            
    return collected

def create_synthetic_certificate(index: int) -> dict:
    """
    Generates a high-resolution synthetic certificate image and PDF with embedded QR code.
    Saves image, PDF, and returns ground-truth metadata.
    """
    os.makedirs(SYNTH_CERT_DIR, exist_ok=True)
    
    width, height = 1800, 2500
    inst_name = random.choice(SAMPLE_INSTITUTIONS)
    cert_title = random.choice(SAMPLE_TITLES)
    recipient = random.choice(SAMPLE_RECIPIENTS)
    course = random.choice(SAMPLE_COURSES)
    issuer_name, issuer_title = random.choice(SAMPLE_ISSUERS)
    date_str = random.choice(SAMPLE_DATES)
    description = random.choice(SAMPLE_DESCRIPTIONS)
    
    # Generate random certificate ID
    hex_part = f"{random.randint(0x1000, 0xFFFF):04X}-{random.randint(0x1000, 0xFFFF):04X}-{random.randint(0x1000, 0xFFFF):04X}"
    cert_id = f"CERT-{hex_part}"

    # Palette styles
    bg_colors = [(255, 255, 255), (252, 252, 250), (255, 254, 245)]
    bg_color = random.choice(bg_colors)
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Outer and inner borders
    border_color = (25, 55, 120) if index % 2 == 0 else (50, 50, 50)
    gold_color = (180, 140, 45)
    
    draw.rectangle([50, 50, width - 50, height - 50], outline=border_color, width=6)
    draw.rectangle([70, 70, width - 70, height - 70], outline=gold_color, width=2)
    draw.rectangle([90, 90, width - 90, height - 90], outline=border_color, width=1)

    # Use default/standard font rendering with scale
    try:
        font_large = ImageFont.truetype("arial.ttf", 64)
        font_med = ImageFont.truetype("arial.ttf", 46)
        font_title = ImageFont.truetype("times.ttf", 72)
        font_name = ImageFont.truetype("times.ttf", 92)
        font_small = ImageFont.truetype("arial.ttf", 36)
        font_tiny = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font_large = font_med = font_title = font_name = font_small = font_tiny = ImageFont.load_default()

    # 1. Certificate ID at top
    id_text = f"CERTIFICATE ID: {cert_id}"
    draw.text((width // 2, 180), id_text, fill=(70, 70, 70), font=font_tiny, anchor="mm")

    # 2. Institution Name
    draw.text((width // 2, 330), inst_name, fill=border_color, font=font_large, anchor="mm")

    # 3. Certificate Title
    draw.text((width // 2, 480), cert_title, fill=(20, 20, 80), font=font_title, anchor="mm")

    # 4. Presentation anchor
    draw.text((width // 2, 880), "Proudly Presented To", fill=(100, 100, 100), font=font_small, anchor="mm")

    # 5. Recipient Name
    draw.text((width // 2, 1040), recipient, fill=(15, 15, 25), font=font_name, anchor="mm")

    # 6. Description
    draw.text((width // 2, 1240), description, fill=(60, 60, 60), font=font_small, anchor="mm")
    draw.text((width // 2, 1340), "in", fill=(100, 100, 100), font=font_small, anchor="mm")

    # 7. Course Name
    draw.text((width // 2, 1520), course, fill=border_color, font=font_large, anchor="mm")

    # 8. Date of Completion
    draw.text((width // 2, 1840), "DATE OF COMPLETION", fill=(110, 110, 110), font=font_tiny, anchor="mm")
    draw.text((width // 2, 1920), date_str, fill=(20, 20, 20), font=font_med, anchor="mm")

    # 9. Signatures and Issuer
    draw.line([(240, 2260), (680, 2260)], fill=(80, 80, 80), width=2)
    draw.text((460, 2300), issuer_name, fill=(10, 10, 10), font=font_med, anchor="mm")
    draw.text((460, 2360), issuer_title, fill=(90, 90, 90), font=font_tiny, anchor="mm")

    # 10. Generate and paste real QR code
    qr_payload = {
        "certificateID": cert_id,
        "recipientName": recipient,
        "courseName": course,
        "institutionName": inst_name,
        "issuerName": issuer_name,
        "completionDate": date_str,
        "certificateTitle": cert_title
    }
    
    qr = qrcode.QRCode(box_size=4, border=1)
    qr.add_data(json.dumps(qr_payload))
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    qr_w, qr_h = qr_img.size
    
    qr_x, qr_y = width - 380, 2140
    img.paste(qr_img, (qr_x, qr_y))
    draw.text((qr_x + qr_w // 2, qr_y + qr_h + 30), "Scan to Verify", fill=(80, 80, 80), font=font_tiny, anchor="mm")

    # Save PNG and PDF
    base_name = f"synth_certificate_{index:02d}"
    img_path = os.path.join(SYNTH_CERT_DIR, f"{base_name}.png")
    pdf_path = os.path.join(SYNTH_CERT_DIR, f"{base_name}.pdf")
    
    img.save(img_path, "PNG")
    img.save(pdf_path, "PDF", resolution=300.0)

    gt_data = {
        "id": base_name,
        "image_file": img_path,
        "pdf_file": pdf_path,
        "ground_truth": {
            "recipientName": recipient,
            "certificateID": cert_id,
            "issuerName": issuer_name,
            "institutionName": inst_name,
            "certificateTitle": cert_title,
            "courseName": course,
            "certificateDescription": description,
            "completionDate": date_str,
            "has_qr": True
        }
    }
    
    return gt_data

def build_benchmark_dataset(num_synthetic: int = 10):
    """
    Builds the full benchmark dataset combining real and synthetic certificates.
    """
    print(f"\n[Dataset Builder] Initializing benchmark dataset at: {DATASET_ROOT}")
    collect_local_certificates()
    
    catalog = []
    
    # 1. Add ground truth for real certificate_soil.pdf
    soil_gt = {
        "id": "certificate_soil",
        "file_path": r"c:\Users\rayne\Downloads\certificate_soil.pdf",
        "type": "pdf",
        "ground_truth": {
            "recipientName": "Vinay Kumar",
            "certificateID": "BDE702D9-BD68-48E9-86F",
            "issuerName": "Ram Lal Yadav",
            "institutionName": "Institute of Indian Civil Structures",
            "certificateTitle": "CERTIFICATE OF COMPLETION",
            "courseName": "Soil Analysis",
            "certificateDescription": "outstanding dedication and successful completion of the prestigious program",
            "completionDate": "October 8, 2026",
            "has_qr": True
        }
    }
    catalog.append(soil_gt)

    # 2. Add ground truth for real certificate.png
    cert_png_gt = {
        "id": "certificate_quiz_mybharat",
        "file_path": r"c:\Users\rayne\Downloads\certificate.png",
        "type": "image",
        "ground_truth": {
            "recipientName": "SACHIN RAY",
            "certificateID": None,
            "issuerName": None,
            "institutionName": "Ministry of Youth Affairs & Sports",
            "certificateTitle": "CERTIFICATE OF PARTICIPATION",
            "courseName": "Viksit Bharat Young Leaders Dialogue (VBYLD) 2026",
            "certificateDescription": "for successfully participating in the Online Quiz on",
            "completionDate": "04 October 2025",
            "has_qr": False
        }
    }
    catalog.append(cert_png_gt)

    # 3. Generate diverse synthetic certificates
    print(f"[Dataset Builder] Generating {num_synthetic} diverse synthetic certificates...")
    for i in range(1, num_synthetic + 1):
        item = create_synthetic_certificate(i)
        # Add both Image and PDF entries to catalog
        catalog.append({
            "id": f"{item['id']}_png",
            "file_path": item["image_file"],
            "type": "image",
            "ground_truth": item["ground_truth"]
        })
        catalog.append({
            "id": f"{item['id']}_pdf",
            "file_path": item["pdf_file"],
            "type": "pdf",
            "ground_truth": item["ground_truth"]
        })

    with open(GROUND_TRUTH_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    print(f"[Dataset Builder] Successfully assembled {len(catalog)} test certificates in benchmark catalog!")
    print(f"[Dataset Builder] Catalog saved to: {GROUND_TRUTH_FILE}")

if __name__ == "__main__":
    build_benchmark_dataset(num_synthetic=8)
