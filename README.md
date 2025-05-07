# HR Bot Resume Assistant

**Streamlit‑based application for HR to parse, standardize, store, and search candidate resumes for relevant keywords/skills.**

---

## 🚀 Overview

This HR Bot provides an end‑to‑end pipeline for managing resumes:

1. **Upload & Process**: Parse PDF or DOCX resumes using a custom LlamaParse‑based parser.
2. **Standardize**: Clean and normalize parsed content into a structured JSON format via Azure OpenAI.
3. **Database Management**: Insert or update resumes in MongoDB, browse, search, and delete records.
4. **Boolean Search Engine**: Perform powerful Boolean searches against your resume collection (AND, OR).
5. **Settings**: Configure API keys and database credentials rapidly from the UI.

The primary entry point is `main.py`, which launches the Streamlit UI with navigation for each feature.

---

## 📁 Project Structure

```
DATA-INGESTION-DEPLOYMENT-MAIN/
├── .streamlit/secrets.toml      # Secure credentials (Azure OpenAI, MongoDB)
├── config.py                   # Generated MongoDB config constants
├── db_manager.py               # MongoDB upsert, query, delete utilities
├── final_retriever.py          # Boolean search parser and Streamlit resume search UI
├── llama_resume_parser.py      # LlamaParse‑based resume parser (PDF/DOCX)
├── standardizer.py             # Azure OpenAI resume standardization logic
├── main.py                     # Main Streamlit application (navigation)
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation
```

---

## 🔧 Prerequisites

* Python 3.8+
* Access to Azure OpenAI (deployment + API key)
* MongoDB instance (URI, database, and collection)
* Llama Cloud API key

---

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/KushagraWadhwaShorthillsAI/V0_HR_Bot_Pipeline
   cd V0_HR_Bot_Pipeline
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure secrets**

   * Populate with your Azure OpenAI keys, MongoDB URI, and Llama Cloud API key in secrets.toml file


---

## 🚀 Running the App

Launch the Streamlit interface:

```bash
python -m streamlit run main.py
```

Open your browser at `http://localhost:8503` and use the sidebar to:

* **Upload & Process**: Step 1) Parse → 2) Standardize → 3) Upload
* **Database Management**: Browse, search, edit, or delete resumes
* **Boolean Search Engine**: Enter Boolean queries (e.g., `Python AND (Django OR Flask)`, `"Machine Learning" AND Python`)
* **Settings**: Update API keys and MongoDB connection on the fly

---

## 🔍 Boolean Search Tips

* **AND**: `JavaScript AND React`
* **OR**: `AWS OR Azure`
* **Grouped logic**: `(Python OR R) AND MachineLearning (Dont give spaces between multi word skills)`

---

## 📦 Module Summaries

* **`llama_resume_parser.py`**:

  * Uses LlamaParse to extract text & hyperlinks from resumes.
  * Supported formats: `.pdf`, `.docx`.

* **`standardizer.py`**:

  * Invokes Azure OpenAI to convert parsed Markdown into a fixed JSON schema.
  * Ensures consistent keys for name, email, education, experience, skills, etc.

* **`db_manager.py`**:

  * Upsert resumes by name/email, bulk insert, find, update, delete.
  * CLI support for file/folder operations.

* **`final_retriever.py`**:

  * Implements BooleanSearchParser (AND, OR), normalizes and flattens JSON.
  * Renders card/table views of matching candidate profiles.

* **`main.py`**:

  * Streamlit navigation across all features.
  * Handles file uploads, asynchronous standardization, progress tracking, and state management.

---
