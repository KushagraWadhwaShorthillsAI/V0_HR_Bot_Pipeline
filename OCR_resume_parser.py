# Ensure this file is correctly imported in main.py
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import re
from pdf2image import convert_from_path
import pytesseract
import cv2
from PIL import Image

class ResumeParserwithOCR:
    def __init__(self):
        load_dotenv()
        self.RESUME_DIR = Path("resumewithdefects")
        self.OUTPUT_DIR = Path("ocr_resumeswithdefects")
        self.SUPPORTED_EXTENSIONS = [".pdf", ".jpg",". jpeg", ".png",".docx"]

    def extract_links_from_text(self, text, file_name):
        """Extract URLs from OCR-parsed text using regex."""
        try:
            url_pattern = r'(https?://[^\s]+|www\.[^\s]+)'
            urls = re.findall(url_pattern, text)
            links = [{"text": url, "uri": url} for url in urls]
            return links
        except Exception as e:
            print(f"⚠️ Failed to extract URLs from {file_name} OCR text: {e}")
            return []

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using OCR."""
        try:
            images = convert_from_path(pdf_path, dpi=300)
            text = ""
            for i, image in enumerate(images):
                text += f"\n\n--- Page {i+1} of {os.path.basename(pdf_path)} ---\n\n"
                text += pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            print(f"❌ Failed to extract text from PDF {pdf_path.name} with OCR: {e}")
            return ""

    def extract_text_from_image(self, image_path):
        """Extract text from image using OCR."""
        try:
            img = cv2.imread(str(image_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pil_img = Image.fromarray(gray)
            text = pytesseract.image_to_string(pil_img)
            return f"\n\n--- Image File: {os.path.basename(image_path)} ---\n\n{text}".strip()
        except Exception as e:
            print(f"❌ Failed to extract text from image {image_path.name} with OCR: {e}")
            return ""

    def parse_resume(self, file_path):
        """Parse resume file using OCR and extract links."""
        try:
            parsed = {"file": file_path.name, "links": []}
            if file_path.suffix.lower() == ".jpg":
                parsed["content"] = self.extract_text_from_image(file_path)
            elif file_path.suffix.lower() == ".pdf":
                parsed["content"] = self.extract_text_from_pdf(file_path)
            else:
                print(f"❌ Unsupported file type: {file_path.name}")
                return None

            parsed["links"] = self.extract_links_from_text(parsed["content"], file_path.name)
            return parsed if parsed["content"] else None
        except Exception as e:
            print(f"❌ Failed to parse {file_path.name}: {e}")
            return None

    def save_to_json(self, data, output_path):
        """Save parsed data to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

