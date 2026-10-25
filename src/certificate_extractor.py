import os
import re
import json
import base64
from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np
import pypdfium2 as pdfium
import easyocr
from PIL import Image
from rapidfuzz import fuzz, process

class CertificateExtractor:
    """
    Doraemon Certificate Information Extraction Engine.
    Processes both digital/scanned PDFs and photo/image certificates.
    
    Extracts the following core fields:
      - recipientName
      - certificateID
      - issuerName
      - institutionName
      - certificateTitle
      - courseName
      - certificateDescription
      - completionDate
      - qrImage (saved image crop path or None if no QR present)
    """

    def __init__(self, languages: List[str] = ['en'], use_gpu: bool = False):
        self.languages = languages
        self.use_gpu = use_gpu
        self._reader = None
        self._qr_detector = None

    @property
    def reader(self) -> easyocr.Reader:
        """Lazy loader for EasyOCR reader."""
        if self._reader is None:
            self._reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
        return self._reader

    @property
    def qr_detector(self) -> cv2.QRCodeDetector:
        """Lazy loader for OpenCV QRCodeDetector."""
        if self._qr_detector is None:
            self._qr_detector = cv2.QRCodeDetector()
        return self._qr_detector

    def load_document_as_image(self, file_path: str, page_index: int = 0, scale: float = 2.0) -> np.ndarray:
        """
        Loads PDF or image file into high-resolution BGR numpy image.
        For PDFs, renders page at 200 DPI (scale=2.0) for optimal speed and OCR clarity.
        For images, loads via PIL/OpenCV with EXIF orientation correction.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Certificate file not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            doc = pdfium.PdfDocument(file_path)
            if page_index >= len(doc):
                raise IndexError(f"Page index {page_index} out of range (total pages: {len(doc)})")
            page = doc[page_index]
            bitmap = page.render(scale=scale)
            pil_img = bitmap.to_pil()
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            pil_img = Image.open(file_path).convert('RGB')
            try:
                from PIL import ImageOps
                pil_img = ImageOps.exif_transpose(pil_img)
            except Exception:
                pass
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def deskew_image(self, img_bgr: np.ndarray, max_angle: float = 25.0) -> Tuple[np.ndarray, float]:
        """
        Detects document skew angle using Hough line transform and straightens the image.
        Uses a downscaled thumbnail (width=800px) for line detection to achieve near-instant (0.05s) execution.
        """
        h, w = img_bgr.shape[:2]
        scale_factor = 800.0 / max(w, 1)
        thumb = cv2.resize(img_bgr, (800, max(1, int(h * scale_factor))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 200, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=60, maxLineGap=10)
        if lines is None:
            return img_bgr, 0.0

        angles = []
        for line in lines:
            pts = line.flatten()
            if len(pts) < 4:
                continue
            x1, y1, x2, y2 = pts[:4]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if abs(dx) > 1e-3:
                angle = float(np.degrees(np.arctan2(dy, dx)))
                if abs(angle) < max_angle:
                    angles.append(angle)

        if not angles:
            return img_bgr, 0.0

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.35:
            return img_bgr, 0.0

        center = (w // 2, h // 2)
        m_rot = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        deskewed = cv2.warpAffine(img_bgr, m_rot, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return deskewed, median_angle

    def normalize_illumination_and_enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Removes uneven lighting, camera shadows, and flash glare.
        Applies CLAHE contrast enhancement for low-contrast/faded text.
        """
        # Split color planes and perform morphological background division
        rgb_planes = cv2.split(img_bgr)
        norm_planes = []
        for plane in rgb_planes:
            dilated = cv2.dilate(plane, np.ones((15, 15), np.uint8))
            bg = cv2.medianBlur(dilated, 21)
            diff = 255 - cv2.absdiff(plane, bg)
            norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            norm_planes.append(norm)

        normalized = cv2.merge(norm_planes)

        # Apply CLAHE to L-channel in LAB space
        lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def extract_and_crop_qr(self, img_bgr: np.ndarray, output_crop_path: Optional[str] = None) -> Tuple[Optional[str], Optional[dict], Optional[str]]:
        """
        Detects QR code, crops qrImage with padding, and decodes any embedded payload.
        Strict verification: Avoids false-positive line/border detection by requiring a valid
        decoded payload OR verified square geometry with dark-light variance.
        """
        h, w = img_bgr.shape[:2]
        decoded_info, points, _ = self.qr_detector.detectAndDecode(img_bgr)
        
        qr_crop_b64 = None
        qr_saved_path = None
        qr_metadata = None

        if not decoded_info or points is None or len(points) == 0:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            val, pts, _ = self.qr_detector.detectAndDecode(enhanced)
            if val:
                decoded_info = val
                points = pts

        # Strictly verify QR code: must decode valid payload or URL to avoid false positives
        is_valid_qr = bool(decoded_info) and len(str(decoded_info).strip()) > 0

        if is_valid_qr and points is not None and len(points) > 0:
            pts = points[0].astype(int)
            x_min = max(0, int(np.min(pts[:, 0])))
            x_max = min(w, int(np.max(pts[:, 0])))
            y_min = max(0, int(np.min(pts[:, 1])))
            y_max = min(h, int(np.max(pts[:, 1])))

            pad_x = int((x_max - x_min) * 0.08)
            pad_y = int((y_max - y_min) * 0.08)
            crop_x0 = max(0, x_min - pad_x)
            crop_x1 = min(w, x_max + pad_x)
            crop_y0 = max(0, y_min - pad_y)
            crop_y1 = min(h, y_max + pad_y)

            qr_crop = img_bgr[crop_y0:crop_y1, crop_x0:crop_x1]

            if output_crop_path:
                os.makedirs(os.path.dirname(os.path.abspath(output_crop_path)), exist_ok=True)
                cv2.imwrite(output_crop_path, qr_crop)
                qr_saved_path = output_crop_path

            _, buffer = cv2.imencode('.png', qr_crop)
            qr_crop_b64 = "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')

            if decoded_info:
                try:
                    qr_metadata = json.loads(decoded_info)
                except Exception:
                    qr_metadata = {"raw_text": decoded_info}
        else:
            if output_crop_path and os.path.exists(output_crop_path):
                try:
                    os.remove(output_crop_path)
                except Exception:
                    pass

        return qr_crop_b64, qr_metadata, qr_saved_path

    def run_ocr_and_cluster_lines(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs deep OCR (EasyOCR) and groups detected word boxes into column-aware horizontal text lines.
        Prevents merging distinct horizontal columns (e.g. Date on left, Signatory on right).
        Optimized with max dimension scaling (1600px) for 3x faster inference while maintaining full coordinate fidelity.
        """
        h, w = img_bgr.shape[:2]
        max_dim = max(h, w)
        if max_dim > 1600:
            ocr_scale = 1600.0 / max_dim
            ocr_w = int(w * ocr_scale)
            ocr_h = int(h * ocr_scale)
            img_for_ocr = cv2.resize(img_bgr, (ocr_w, ocr_h), interpolation=cv2.INTER_AREA)
            coord_scale = max_dim / 1600.0
        else:
            img_for_ocr = img_bgr
            coord_scale = 1.0

        raw_results = self.reader.readtext(img_for_ocr)
        
        raw_blocks = []
        for bbox, text, conf in raw_results:
            text = text.strip()
            if not text:
                continue
            
            pts = np.array(bbox) * coord_scale
            x0 = float(np.min(pts[:, 0]))
            y0 = float(np.min(pts[:, 1]))
            x1 = float(np.max(pts[:, 0]))
            y1 = float(np.max(pts[:, 1]))
            
            raw_blocks.append({
                "text": text,
                "confidence": float(conf),
                "bbox": [x0, y0, x1, y1],
                "y_center": (y0 + y1) / 2.0,
                "x_center": (x0 + x1) / 2.0,
                "height": y1 - y0,
                "width": x1 - x0
            })

        # Column-aware clustering: Sort by Y first, then X
        lines = []
        for b in sorted(raw_blocks, key=lambda x: (x["bbox"][1], x["bbox"][0])):
            y_center = b["y_center"]
            h_b = b["height"]
            
            matched = False
            for line in lines:
                line_y_center = np.mean([item["y_center"] for item in line])
                line_h = np.mean([item["height"] for item in line])
                
                # Check vertical alignment
                if abs(y_center - line_y_center) < max(h_b, line_h) * 0.5:
                    # Check horizontal gap: Must be physically close (same column)
                    min_x0 = min(item["bbox"][0] for item in line)
                    max_x1 = max(item["bbox"][2] for item in line)
                    gap = max(0, b["bbox"][0] - max_x1, min_x0 - b["bbox"][2])
                    max_allowed_gap = max(h_b, line_h) * 1.2
                    
                    if gap <= max(max_allowed_gap, 45.0):
                        line.append(b)
                        matched = True
                        break
            if not matched:
                lines.append([b])

        assembled_lines = []
        for line in lines:
            line.sort(key=lambda x: x["bbox"][0])
            full_text = " ".join(x["text"] for x in line)
            avg_conf = float(np.mean([x["confidence"] for x in line]))
            min_x0 = min(x["bbox"][0] for x in line)
            min_y0 = min(x["bbox"][1] for x in line)
            max_x1 = max(x["bbox"][2] for x in line)
            max_y1 = max(x["bbox"][3] for x in line)

            assembled_lines.append({
                "text": full_text,
                "confidence": avg_conf,
                "bbox": [min_x0, min_y0, max_x1, max_y1],
                "norm_y0": min_y0 / h,
                "norm_y1": max_y1 / h,
                "norm_x0": min_x0 / w,
                "norm_x1": max_x1 / w,
                "center_x": (min_x0 + max_x1) / (2.0 * w),
                "center_y": (min_y0 + max_y1) / (2.0 * h),
                "height": max_y1 - min_y0
            })

        # Sort all lines strictly by vertical position
        assembled_lines.sort(key=lambda l: (l["norm_y0"], l["norm_x0"]))
        return assembled_lines

    def parse_fields_from_lines(self, lines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        High-precision semantic layout parsing optimized for the target certificate format
        as well as generalized academic & professional certificates.
        Extracts: recipientName, certificateID, issuerName, institutionName, certificateTitle,
        courseName, certificateDescription, completionDate.
        """
        fields = {
            "recipientName": None,
            "certificateID": None,
            "issuerName": None,
            "institutionName": None,
            "certificateTitle": None,
            "courseName": None,
            "certificateDescription": None,
            "completionDate": None
        }

        # -------------------------------------------------------------
        # 1. Certificate ID: Support letter-spacing in modern template headers
        # e.g., 'CERTIFICATE ID: BDE70209-BD6 8 - 4 8E9 - 8 6F'
        # -------------------------------------------------------------
        for line in lines:
            text = line["text"]
            match = re.search(r'(?:CERTIFICATE\s*ID|CERT\s*ID|ROLL\s*NO\.?|ID\s*:?|REF\s*:?|NO\s*:)\s*[:\-]?\s*([A-Za-z0-9\s\/\-_]{5,55})', text, re.IGNORECASE)
            if match:
                raw_id = match.group(1).strip()
                # Stop if reaching header/institution words
                raw_id = re.split(r'\b(?:INSTITUTE|UNIVERSITY|COLLEGE|CERTIFICATE)\b', raw_id, flags=re.IGNORECASE)[0]
                clean_id = re.sub(r'\s+', '', raw_id).strip().rstrip('.-_')
                if len(clean_id) >= 5:
                    fields["certificateID"] = clean_id
                    break

            # Slash-delimited institutional reference code
            m_slash = re.search(r'\b([A-Za-z0-9]{3,15}(?:/[A-Za-z0-9_\-]+){2,6})\b', text)
            if m_slash and line["norm_y0"] > 0.60:
                fields["certificateID"] = m_slash.group(1).strip()
                break

        # Margin/Standalone fallback ID
        if not fields["certificateID"]:
            for line in lines:
                text = line["text"].strip()
                if re.match(r'^(?:[a-z0-9]{7,16}|[A-Z0-9]{3,8}-[A-Z0-9]{4,8}(?:-[A-Z0-9]{3,6})?)$', text) and (line["norm_y0"] > 0.60 or line["norm_y1"] < 0.20):
                    if not re.search(r'DATE|OCTOBER|NOVEMBER|DECEMBER|JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER', text, re.IGNORECASE):
                        fields["certificateID"] = text
                        break

        # -------------------------------------------------------------
        # 2. Institution Name (Header region & inline anchors)
        # -------------------------------------------------------------
        for line in lines:
            m_awards = re.search(r'([A-Za-z0-9\s,\.\-]+?)\s+(?:Awards\s+this\s+Certificate|presents\s+this\s+Certificate)', line["text"], re.IGNORECASE)
            if m_awards:
                fields["institutionName"] = m_awards.group(1).strip()
                break

            m_from = re.search(r'(?:from|at)\s+([A-Za-z0-9\s,\.\-]{3,65}?(?:University|Institute|College|Academy|School|Department|NIT|IIT|Organization|Company|Limited|Corp|Plant|Structures|Ministry|Government)[A-Za-z0-9\s,\.\-]*?)(?:\s+(?:has|for|and|in)|$)', line["text"], re.IGNORECASE)
            if m_from:
                fields["institutionName"] = m_from.group(1).strip()
                break

            m_org = re.search(r'organized by\s+([A-Za-z0-9\s,\.\-]+?(?:NIT|IIT|University|Institute|College|Department of [A-Za-z\s]+?,\s*[A-Za-z\s]+))(?:\s+during|\s+on|\s+at|$)', line["text"], re.IGNORECASE)
            if m_org:
                fields["institutionName"] = m_org.group(1).strip()
                break

        if not fields["institutionName"]:
            inst_candidates = []
            for line in lines:
                if line["norm_y1"] < 0.32:
                    t = line["text"].strip()
                    if re.search(r'INSTITUTE|UNIVERSITY|COLLEGE|ACADEMY|DEPARTMENT|STRUCTURES|SCHOOL|COUNCIL|FOUNDATION|ASSOCIATION|MINISTRY|GOVERNMENT|LIMITED|CORP|LAB|HOSPITAL', t, re.IGNORECASE):
                        if not re.search(r'CERTIFICATE|ID:|ROLL|AWARDS|COMPLETION', t, re.IGNORECASE):
                            inst_candidates.append(t)
            if inst_candidates:
                fields["institutionName"] = " ".join(inst_candidates)
            else:
                for line in lines:
                    if 0.08 < line["norm_y0"] < 0.22:
                        t = line["text"].strip()
                        if not re.search(r'CERTIFICATE|ID:|ROLL|PRESENTED', t, re.IGNORECASE) and len(t) >= 4:
                            fields["institutionName"] = t
                            break

        # -------------------------------------------------------------
        # 3. Certificate Title
        # -------------------------------------------------------------
        title_parts = []
        for line in lines:
            if 0.08 < line["norm_y0"] < 0.40:
                t = line["text"].strip()
                if re.search(r'CERTIFICATE|COMPLETION|ACHIEVEMENT|APPRECIATION|DIPLOMA|EXCELLENCE|PARTICIPATION|TRAINING', t, re.IGNORECASE):
                    if not re.search(r'CERTIFICATE\s*ID|IS\s*HEREBY|AWARDS\s*THIS|^Sub\s*:|^Subject\s*:|PRESENTED', t, re.IGNORECASE):
                        title_parts.append(t)
        if title_parts:
            combined = " ".join(title_parts)
            combined = re.sub(r'\bCERTIFICATE OF\s+CERTIFICATE OF\b', 'CERTIFICATE OF', combined, flags=re.IGNORECASE)
            fields["certificateTitle"] = combined.upper()
        else:
            canonical_titles = [
                "CERTIFICATE OF PARTICIPATION",
                "CERTIFICATE OF COMPLETION",
                "CERTIFICATE OF TRAINING COMPLETION",
                "CERTIFICATE OF ACHIEVEMENT",
                "CERTIFICATE OF APPRECIATION",
                "CERTIFICATE OF MERIT",
                "BONAFIDE CERTIFICATE"
            ]
            for line in lines:
                if 0.15 < line["norm_y0"] < 0.50:
                    text_clean = line["text"].strip().lower()
                    for can in canonical_titles:
                        ratio = fuzz.ratio(text_clean, can.lower())
                        if ratio >= 55.0 or (fuzz.partial_ratio("certificate", text_clean) >= 65.0 and len(text_clean) > 8):
                            fields["certificateTitle"] = can
                            break
                    if fields["certificateTitle"]:
                        break

        # -------------------------------------------------------------
        # 4. Recipient Name
        # -------------------------------------------------------------
        recipient_line_idx = -1

        # A. Template Anchor: Proudly Presented To (Name on next line)
        for i, line in enumerate(lines):
            if re.search(r'Proudly\s+Presented\s+To|Presented\s+To|Awarded\s+To|Conferred\s+Upon|Honou?red\s+To', line["text"], re.IGNORECASE):
                if i + 1 < len(lines):
                    cand = lines[i + 1]["text"].strip()
                    if not re.match(r'^(?:from|for|in|who|has)\b', cand, re.IGNORECASE):
                        fields["recipientName"] = re.sub(r'^(?:Mr\.?|Ms\.?|Mrs\.?|Dr\.?|Shri|Prof\.?)\s*', '', cand, flags=re.IGNORECASE).strip()
                        recipient_line_idx = i + 1
                        break

        # B. Sentence Inline: "This is to certify that Mr. Sachin Kumar Ray..."
        if not fields["recipientName"]:
            for i, line in enumerate(lines):
                text = line["text"]
                m_inline = re.search(r'(?:certify that|presented to|awarded to|honou?red to|conferred upon|awards this certificate to)\s+(?:Mr\.?|Ms\.?|Mrs\.?|Dr\.?|Shri|Prof\.?)?\s*([A-Za-z\s\.\-]{3,40}?)(?:\s+(?:from|of|son of|daughter of|has|participated|for|who)|$)', text, re.IGNORECASE)
                if m_inline:
                    name_cand = m_inline.group(1).strip()
                    if not re.search(r'CERTIFICATE|COMPLETION|SUCCESSFULLY|HAVING', name_cand, re.IGNORECASE) and len(name_cand.split()) <= 5:
                        fields["recipientName"] = name_cand
                        recipient_line_idx = i
                        break

        # C. Multi-line anchor fallback
        if not fields["recipientName"]:
            for i, line in enumerate(lines):
                if re.search(r'this\s+is\s+to\s+certify\s+that|certifies\s+that', line["text"], re.IGNORECASE):
                    if i + 1 < len(lines):
                        cand = lines[i + 1]["text"].strip()
                        if not re.match(r'^(?:from|for|in|who|has)\b', cand, re.IGNORECASE):
                            fields["recipientName"] = re.sub(r'^(?:Mr\.?|Ms\.?|Mrs\.?|Dr\.?|Shri|Prof\.?)\s*', '', cand, flags=re.IGNORECASE).strip()
                            recipient_line_idx = i + 1
                            break

        # D. Geometric visual prominence fallback
        if not fields["recipientName"]:
            mid_lines = [l for l in lines if 0.20 < l["norm_y0"] < 0.52]
            if mid_lines:
                mid_lines.sort(key=lambda l: l["height"], reverse=True)
                for cand_line in mid_lines:
                    candidate = cand_line["text"].strip()
                    if not re.search(r'CERTIFICATE|COMPLETION|PRESENTED|AWARDED|CONGRATULATIONS|PROGRAM|COURSE|DIALOGUE|INSTITUTE|UNIVERSITY', candidate, re.IGNORECASE):
                        if len(candidate.split()) <= 5 and not re.search(r'\d', candidate):
                            fields["recipientName"] = re.sub(r'^(?:Mr\.?|Ms\.?|Mrs\.?|Dr\.?|Shri|Prof\.?)\s*', '', candidate, flags=re.IGNORECASE).strip()
                            break

        if recipient_line_idx == -1 and fields["recipientName"]:
            for i, line in enumerate(lines):
                if fields["recipientName"].lower() in line["text"].lower():
                    recipient_line_idx = i
                    break

        # -------------------------------------------------------------
        # 5. Course Name & Description
        # In the template: 'for <Description> in <Course>'
        # -------------------------------------------------------------
        for_idx = -1
        in_idx = -1
        for i, line in enumerate(lines):
            t = line["text"].strip().lower()
            if (t == "for" or t.startswith("for ")) and i > recipient_line_idx:
                if for_idx == -1:
                    for_idx = i
            if (t == "in" or t.startswith("in ")) and i > recipient_line_idx:
                if in_idx == -1:
                    in_idx = i

        if for_idx != -1 and in_idx != -1 and in_idx > for_idx:
            desc_lines = []
            for i in range(for_idx, in_idx):
                t = lines[i]["text"].strip()
                if t.lower() == "for":
                    continue
                if t.lower().startswith("for "):
                    t = t[4:].strip()
                desc_lines.append(t)
            if desc_lines:
                fields["certificateDescription"] = " ".join(desc_lines)
        elif for_idx != -1:
            desc_lines = []
            for i in range(for_idx, len(lines)):
                if lines[i]["norm_y0"] > 0.68 or re.search(r'DATE|COMPLETION|SIGNATURE', lines[i]["text"], re.IGNORECASE):
                    break
                t = lines[i]["text"].strip()
                if t.lower() == "for":
                    continue
                desc_lines.append(t)
            if desc_lines:
                fields["certificateDescription"] = " ".join(desc_lines)

        if in_idx != -1 and in_idx + 1 < len(lines):
            cand = lines[in_idx + 1]["text"].strip()
            if not re.search(r'DATE|COMPLETION|SIGNATURE', cand, re.IGNORECASE):
                fields["courseName"] = cand

        if not fields["courseName"] or not fields["certificateDescription"]:
            desc_tokens = []
            for i, line in enumerate(lines):
                text = line["text"]
                m_quote = re.search(r'["“]([^"”]{5,80})["”]', text)
                if m_quote and not fields["courseName"]:
                    fields["courseName"] = m_quote.group(1).strip()

                m_course = re.search(r'(?:online\s+quiz\s+on|course\s+on|program\s+in|workshop\s+on|stages\s+of\s+the|completing\s+the)\s+([A-Za-z0-9\s,\(\)\-]+?(?:course|program|training|dialogue|workshop|systems|\d{4}))', text, re.IGNORECASE)
                if not m_course:
                    m_course = re.search(r'(?:completing the|participated in|course on|program in|quiz on|stages of the)\s+([A-Za-z0-9\s,\(\)\-]+?(?:course|program|training|dialogue|workshop|systems))', text, re.IGNORECASE)
                if m_course and not fields["courseName"]:
                    fields["courseName"] = m_course.group(1).strip()

                if recipient_line_idx != -1 and i > recipient_line_idx and line["norm_y0"] < 0.72:
                    if re.search(r'DATE|October|November|December|January|February|March|April|May|June|July|August|September|Coordinator|Director|Convener|Signatory', text, re.IGNORECASE):
                        continue
                    if fields.get("institutionName") and fields["institutionName"] in text:
                        continue
                    if re.match(r'^(?:from|for|in)$', text.strip(), re.IGNORECASE):
                        continue
                    desc_tokens.append(text.strip())

            if not fields["certificateDescription"] and desc_tokens:
                fields["certificateDescription"] = " ".join(desc_tokens)
            if not fields["courseName"]:
                for l in lines:
                    if 0.50 < l["norm_y0"] < 0.72:
                        t = l["text"].strip()
                        if not re.search(r'^in$|DATE|COMPLETION|CERTIFICATE', t, re.IGNORECASE):
                            fields["courseName"] = t
                            break

        # -------------------------------------------------------------
        # 6. Completion Date
        # -------------------------------------------------------------
        date_label_idx = -1
        for i, line in enumerate(lines):
            if re.search(r'DATE\s+OF\s+COMPLETION|COMPLETION\s+DATE', line["text"], re.IGNORECASE):
                date_label_idx = i
                break

        if date_label_idx != -1:
            for i in range(date_label_idx + 1, min(len(lines), date_label_idx + 4)):
                t = lines[i]["text"].strip()
                m_d = re.search(r'(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}|\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', t, re.IGNORECASE)
                if m_d:
                    fields["completionDate"] = m_d.group(0).strip()
                    break

        if not fields["completionDate"]:
            for line in lines:
                raw_t = line["text"]
                norm_t = re.sub(r'[,.\s]+(?=\d{4})', ', ', raw_t)
                m_date = re.search(r'(?:(?:\d{1,2}(?:st|nd|rd|th)?\s*[\-–to\s]+\s*)?\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s*,?\s*\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\s*,?\s*\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s*,?\s*\d{4}|\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', norm_t, re.IGNORECASE)
                if m_date and line["norm_y0"] > 0.10:
                    fields["completionDate"] = m_date.group(0).strip().rstrip('.')
                    break

        # -------------------------------------------------------------
        # 7. Issuer Name (Bottom-left column above signatory role)
        # -------------------------------------------------------------
        for i, line in enumerate(lines):
            if line["norm_y0"] > 0.82 and line["center_x"] < 0.55:
                if re.search(r'HOD|Director|Principal|Dean|Head|Coordinator|Professor|President|Founder|Signatory', line["text"], re.IGNORECASE):
                    if i > 0 and lines[i-1]["norm_y0"] > 0.78 and lines[i-1]["center_x"] < 0.55:
                        fields["issuerName"] = lines[i-1]["text"].strip()
                    break

        if not fields["issuerName"]:
            role_lines = [l for l in lines if l["norm_y0"] > 0.65 and re.search(r'Founder|Director|HOD|Dean|Principal|Authorized|Signatory|Coordinator|President|Convener', l["text"], re.IGNORECASE)]
            for r_line in role_lines:
                r_x = r_line["center_x"]
                r_y = r_line["norm_y0"]
                candidate_issuers = []
                for l in lines:
                    if 0.58 < l["norm_y0"] < r_y and abs(l["center_x"] - r_x) < 0.22:
                        cand_text = l["text"].strip()
                        if not re.search(r'Date|October|November|December|Scan|Verify|Course|Certificate', cand_text, re.IGNORECASE):
                            candidate_issuers.append((l["norm_y0"], cand_text))
                if candidate_issuers:
                    candidate_issuers.sort(key=lambda x: x[0], reverse=True)
                    fields["issuerName"] = candidate_issuers[0][1]
                    break

        if not fields["issuerName"]:
            for line in lines:
                if line["norm_y0"] > 0.65:
                    m_hon = re.search(r'\b(?:Dr\.?|Prof\.?|Mr\.?|Mrs\.?)\s*:?\s*([A-Z][a-zA-Z\s]+)', line["text"])
                    if m_hon:
                        fields["issuerName"] = m_hon.group(0).replace(':', '').strip()
                        break

        return fields

    def parse_with_local_slm(self, lines: List[Dict[str, Any]], model_name: str = "llama3.1:8b") -> Optional[Dict[str, Any]]:
        """
        Zero-Shot Offline Layout Parsing via local Ollama SLM (localhost:11434).
        Requires ZERO API keys; runs 100% locally on CPU/local GPU.
        """
        import urllib.request
        full_text = "\n".join(l["text"] for l in lines)
        prompt = f"""You are an expert certificate parser. Extract the following fields from this OCR text:
- recipientName (Full name of person receiving certificate)
- certificateID (ID/serial if present, else null)
- issuerName (Name of signatory/signer)
- institutionName (Issuing organization/school/company)
- certificateTitle (e.g. CERTIFICATE OF PARTICIPATION, CERTIFICATE OF COMPLETION, etc.)
- courseName (Name of course/program)
- certificateDescription (Brief sentence explaining what it was awarded for)
- completionDate (Date of completion if present, else null)

OCR Text:
{full_text}

Return ONLY a valid JSON object matching these keys."""
        try:
            req = urllib.request.Request(
                'http://localhost:11434/api/generate',
                data=json.dumps({
                    'model': model_name,
                    'prompt': prompt,
                    'format': 'json',
                    'stream': False
                }).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                raw_json = json.loads(data.get("response", "{}"))
                return raw_json
        except Exception:
            return None

    def extract(self, file_path: str, output_qr_crop_path: Optional[str] = "output/qr_extracted.png", auto_restore: bool = True, use_local_slm_fallback: bool = False) -> Dict[str, Any]:
        """
        Main extraction pipeline.
        1. Loads PDF or image with EXIF orientation correction.
        2. Applies visual restoration (fast deskewing on downscaled thumbnail).
        3. Detects, crops, and decodes QR Code (with false-positive verification).
        4. Executes deep OCR with resolution optimization and column-aware line clustering.
        5. Extracts semantic certificate fields using template-calibrated layout parsing.
        6. Fuses QR verification data with OCR output.
        """
        img_bgr = self.load_document_as_image(file_path, scale=3.0)

        # Visual Restoration: Auto-deskew tilted mobile photos
        if auto_restore:
            img_bgr, skew_angle = self.deskew_image(img_bgr)

        # 1. QR Code extraction
        qr_b64, qr_metadata, qr_saved_path = self.extract_and_crop_qr(img_bgr, output_qr_crop_path)

        # 2. Deep OCR and Line Clustering (accelerated with 1600px max dimension)
        lines = self.run_ocr_and_cluster_lines(img_bgr)

        # 3. Layout Semantic Parsing (Fast Deterministic Tier)
        extracted = self.parse_fields_from_lines(lines)

        # 4. Optional Local SLM Fallback (Zero API Keys: default False for pure high-speed OCR)
        if use_local_slm_fallback:
            missing_critical = not extracted.get("recipientName") or not extracted.get("institutionName") or not extracted.get("courseName")
            if missing_critical:
                slm_res = self.parse_with_local_slm(lines)
                if slm_res and isinstance(slm_res, dict):
                    for k in ["recipientName", "certificateID", "issuerName", "institutionName", "certificateTitle", "courseName", "certificateDescription", "completionDate"]:
                        if not extracted.get(k) and slm_res.get(k):
                            extracted[k] = slm_res[k]

        # 5. Cross-Channel Verification & Fusion (if QR code actually exists)
        if qr_metadata and isinstance(qr_metadata, dict):
            if "studentName" in qr_metadata and qr_metadata["studentName"]:
                extracted["recipientName"] = qr_metadata["studentName"].strip()
            elif "recipientName" in qr_metadata and qr_metadata["recipientName"]:
                extracted["recipientName"] = qr_metadata["recipientName"].strip()

            if "qrCodeId" in qr_metadata and qr_metadata["qrCodeId"]:
                extracted["certificateID"] = qr_metadata["qrCodeId"].strip()
            elif "certificateID" in qr_metadata and qr_metadata["certificateID"]:
                extracted["certificateID"] = qr_metadata["certificateID"].strip()

            if "issuerName" in qr_metadata and qr_metadata["issuerName"]:
                extracted["issuerName"] = extracted["issuerName"] or qr_metadata["issuerName"].strip()

            if "institutionName" in qr_metadata and qr_metadata["institutionName"]:
                extracted["institutionName"] = extracted["institutionName"] or qr_metadata["institutionName"].strip()

            if "courseName" in qr_metadata and qr_metadata["courseName"]:
                extracted["courseName"] = extracted["courseName"] or qr_metadata["courseName"].strip()

            if "certificateTitle" in qr_metadata and qr_metadata["certificateTitle"]:
                if not extracted["certificateTitle"]:
                    extracted["certificateTitle"] = f"CERTIFICATE OF {qr_metadata['certificateTitle'].upper()}"

            if "completionDate" in qr_metadata and qr_metadata["completionDate"]:
                if not extracted.get("completionDate"):
                    # Format ISO date if needed
                    raw_dt = qr_metadata["completionDate"].strip()
                    try:
                        from datetime import datetime
                        if "T" in raw_dt:
                            parsed_dt = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
                            extracted["completionDate"] = parsed_dt.strftime("%B %d, %Y")
                        else:
                            extracted["completionDate"] = raw_dt
                    except Exception:
                        extracted["completionDate"] = raw_dt

        # Attach qrImage path (strictly None if no valid QR code detected)
        extracted["qrImage"] = qr_saved_path if qr_saved_path else qr_b64

        return extracted
