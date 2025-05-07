import os
import json
from pathlib import Path
import httpx
import asyncio
import streamlit as st  # Added for secrets access

class ResumeStandardizer:
    def __init__(self):
        self.api_key = st.secrets["azure_openai"]["api_key"]
        self.endpoint = st.secrets["azure_openai"]["endpoint"]
        self.deployment = st.secrets["azure_openai"]["deployment"]
        self.api_version = st.secrets["azure_openai"].get("api_version", "2024-08-01-preview")

        if not self.api_key or not self.endpoint or not self.deployment:
            raise ValueError("❌ Missing Azure OpenAI secrets in secrets.toml")

        self.INPUT_DIR = Path("data2/llama_parse_resumes")
        self.OUTPUT_DIR = Path("data2/standardized_resumes")
        self.RAW_LOG_DIR = Path("data2/standardized_raw_responses")
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)

    def make_standardizer_prompt(self, content: str, links: list) -> str:
        return self._prompt_template(content, links)
    
    def _prompt_template(self, content, links):
        return f"""
You are an intelligent and robust context-aware resume standardizer. Your task is to convert resume content into a clean, structured, normalized/standardized JSON format suitable for both semantic retrieval and relational database storage.

--- PARSED RESUME CONTENT ---
The content has been parsed using *LlamaParse*, so please note:
- It is formatted in markdown-style text.
- Section names like Education, Experience, Projects, Skills, etc., are likely to be marked by headings or bullet points.
- The resume templates can vary significantly. For example, "Professional Experience", "Work History", or "Employment" may all refer to the same section. We need you for Normalizing such variants into the defined structure mentioned below.
- Some artifacts or repetition may occur due to Llama Parse not being able to understand the text is embedded link.

--- EXTRACTED HYPERLINKS ---
The hyperlinks have been extracted using *Fitz*. Please note:
- The links are mostly accurate, but the anchor texts may be confusing and should not be blindly trusted.
- Only use a link where the surrounding content makes the intent clear (e.g., GitHub → social, certificate → certifications).

--- RESUME CONTENT ---
\"\"\"{content}\"\"\"

--- HYPERLINKS (Extracted from PDF) ---
{json.dumps(links, indent=2)}

--- STANDARDIZED STRUCTURE ---
Convert the resume to a JSON object strictly following this structure:

{{
  "name": str,
  "email": str,
  "phone": str,
  "location": str,
  "summary": str,
  "education": [
    {{
      "degree": str,
      "institution": str,
      "year": int or str
    }}
  ],
  "experience": [
    {{
      "title": str,
      "company": str,
      "duration": str,
      "location": str (optional),
      "description": str
    }}
  ],
  "skills": [str],
  "projects": [
    {{
      "title": str,
      "description": str,
      "link": str (optional)
    }}
  ],
  "certifications": [
    {{
      "title": str,
      "issuer": str (optional),
      "year": str or int (optional),
      "link": str (only if it can be reliably mapped from extracted hyperlinks)
    }}
  ],
  "languages": [str],
  "social_profiles": [
    {{
      "platform": str,
      "link": str
    }}
  ]
}}

--- GUIDELINES ---
- Strictly follow the above structure. Do not introduce or remove fields arbitrarily.
- Do not change the structure based on data availability. Keep all keys present, with empty arrays if needed.
- Avoid assumptions. Only use data present in the parsed content or reliably inferred via links.
- Ensure all meaningful content is preserved — even if duplicated or malformed in parsing.
- Output only a valid JSON object. Do not wrap in markdown or include any extra commentary.

The quality of your standardization directly impacts downstream processing. Ensure robustness and consistency in formatting across all resumes.
"""

    def clean_llm_response(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```json") and cleaned.endswith("```"):
            return cleaned[7:-3].strip()
        elif cleaned.startswith("```") and cleaned.endswith("```"):
            return cleaned[3:-3].strip()
        return cleaned

    async def call_azure_llm(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }
        body = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that formats resumes into structured JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 6000,
        }
        url = f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def standardize_resume(self, file_path: Path):
        output_path = self.OUTPUT_DIR / file_path.name
        raw_log_path = self.RAW_LOG_DIR / file_path.name.replace(".json", ".md")

        if output_path.exists():
            print(f"⏩ Skipping {file_path.name} (already standardized)")
            return

        with open(file_path, encoding="utf-8") as f:
            raw = json.load(f)

        content = raw.get("content", "")
        links = raw.get("links", [])

        if not content.strip():
            print(f"⚠️ Empty content in {file_path.name}, skipping.")
            return

        prompt = self.make_standardizer_prompt(content, links)

        try:
            print(f"🔍 Standardizing: {file_path.name}")
            raw_response = await self.call_azure_llm(prompt)

            with open(raw_log_path, "w", encoding="utf-8") as f:
                f.write(raw_response)

            cleaned_json = self.clean_llm_response(raw_response)
            parsed_json = json.loads(cleaned_json)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, indent=2, ensure_ascii=False)

            print(f"✅ Saved standardized resume: {output_path.name}")
        except Exception as e:
            print(f"❌ Failed to standardize {file_path.name}: {e}")

    async def run(self):
        files = list(self.INPUT_DIR.glob("*.json"))
        print(f"📂 Found {len(files)} resumes to standardize.\n")
        for file in files:
            await self.standardize_resume(file)