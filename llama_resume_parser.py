# llama_resume_parser.py

import os
import json
from pathlib import Path
import streamlit as st  # Added for secrets access
from llama_parse import LlamaParse
import fitz  # PyMuPDF

class ResumeParser:
    def __init__(self):
        api_key = st.secrets["llama_cloud"]["api_key"]
        if not api_key:
            raise ValueError("❌ LLAMA_CLOUD_API_KEY is missing from secrets.toml")

        self.parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            do_not_unroll_columns=True
        )
        self.SUPPORTED_EXTENSIONS = [".pdf", ".docx"]

    def extract_links_with_fitz(self, file_path):
        links = []
        try:
            with fitz.open(file_path) as doc:
                for page_number, page in enumerate(doc, start=1):
                    for link in page.get_links():
                        if "uri" in link:
                            links.append({
                                "text": page.get_textbox(link["from"]).strip(),
                                "uri": link["uri"]
                            })
        except Exception as e:
            print(f"⚠️ Failed to extract links from {file_path.name}: {e}")
        return links

    def parse_resume(self, file_path):
        try:
            documents = self.parser.load_data(file_path)
            combined_text = "\n".join([doc.text for doc in documents])
            parsed = {
                "file": os.path.basename(file_path),
                "content": combined_text
            }
            if file_path.endswith(".pdf"):
                parsed["links"] = self.extract_links_with_fitz(file_path)
            return parsed
        except Exception as e:
            print(f"❌ Failed to parse {file_path}: {e}")
            return None
