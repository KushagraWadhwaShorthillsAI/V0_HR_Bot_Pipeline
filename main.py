import streamlit as st
import os
import json
import tempfile
from pathlib import Path
import asyncio
from datetime import datetime

# Import your existing modules
from llama_resume_parser import ResumeParser
from standardizer import ResumeStandardizer
from db_manager import ResumeDBManager
from OCR_resume_parser import ResumeParserwithOCR
from final_retriever import run_retriever  # Retriever engine

# Set page configuration
st.set_page_config(
    page_title="HR Resume Bot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for app navigation and explanations
st.sidebar.title("HR Bot Feature Suite")
page = st.sidebar.selectbox("Navigate", [
    "Upload & Process", 
    "Database Management", 
    "Resume Search Engine",
])

# Add explanatory text based on selected page
if page == "Upload & Process":
    st.sidebar.markdown("""
    ### 📤 Upload & Process
    This page allows you to:
    1. Upload multiple resume files (PDF/DOCX)
    2. Automatically process them through our pipeline:
       - Parse and extract content
       - Standardize the format
       - Store in database
    3. Preview processed resumes
    """)
elif page == "Database Management":
    st.sidebar.markdown("""
    ### 💾 Database Management
    This page enables you to:
    1. View all stored resumes
    2. Search resumes by specific fields
    3. View detailed resume information
    4. Manage the resume database
    """)
elif page == "Resume Search Engine":
    st.sidebar.markdown("""
    ### 🔍 Resume Search Engine
    This page provides:
    1. Advanced search capabilities
    2. Boolean search queries
    3. Filter and sort options
    4. Detailed resume matching
    """)

# Initialize session state for tracking job progress
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "standardizing_complete" not in st.session_state:
    st.session_state.standardizing_complete = False
if "db_upload_complete" not in st.session_state:
    st.session_state.db_upload_complete = False
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "standardized_files" not in st.session_state:
    st.session_state.standardized_files = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


def process_uploaded_files(uploaded_files):
    """Process uploaded resume files through the parser"""
    st.session_state.processing_complete = False
    st.session_state.standardizing_complete = False
    st.session_state.db_upload_complete = False
    st.session_state.processed_files = []

    st.write(f"Processing {len(uploaded_files)} files...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    total_files = len(uploaded_files)
    processed_count = 0

    files_to_process = list(uploaded_files)

    for i, uploaded_file in enumerate(files_to_process):
        file_name = uploaded_file.name
        status_text.text(f"Processing {i+1}/{total_files}: {file_name}")

        temp_file_path = os.path.join(temp_dir, file_name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        file_ext = Path(file_name).suffix.lower()
        parser = ResumeParser()  # Instantiate inside loop to avoid event loop issues

        if file_ext not in parser.SUPPORTED_EXTENSIONS:
            st.warning(f"Skipping {file_name}: Unsupported file type {file_ext}")
            continue

        try:
            parsed_resume = parser.parse_resume(temp_file_path)
            if parsed_resume:
                parsed_resume["timestamp"] = datetime.now().isoformat()
                parsed_resume["original_filename"] = file_name

                output_path = parsed_dir / f"{Path(file_name).stem}.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(parsed_resume, f, indent=2, ensure_ascii=False)

                st.session_state.processed_files.append(output_path)
                processed_count += 1
            else:
                st.warning(f"No content extracted from {file_name}")
        except Exception as e:
            st.error(f"Error parsing {file_name}: {str(e)}")

        progress_bar.progress((i + 1) / total_files)

    status_text.text(f"✅ Processed {processed_count}/{total_files} files")
    st.session_state.processing_complete = True
    
async def standardize_resumes():
    """Standardize the parsed resumes using ResumeStandardizer"""
    st.session_state.standardizing_complete = False
    st.session_state.standardized_files = []
    
    # Show status
    st.write(f"Standardizing {len(st.session_state.processed_files)} files...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize the standardizer
    try:
        standardizer = ResumeStandardizer()
    except ValueError as e:
        st.error(f"Error initializing standardizer: {e}")
        return
    
    total_files = len(st.session_state.processed_files)
    standardized_count = 0
    
    # Create a copy of processed files list to prevent any potential issues with iteration
    files_to_standardize = list(st.session_state.processed_files)
    
    for i, file_path in enumerate(files_to_standardize):
        status_text.text(f"Standardizing {i+1}/{total_files}: {file_path.name}")
        
        # Modify the standardizer to use our temp paths
        output_path = standardized_dir / file_path.name
        
        if output_path.exists():
            st.session_state.standardized_files.append(output_path)
            standardized_count += 1
            progress_bar.progress((i + 1) / total_files)
            continue
        
        try:
            with open(file_path, encoding="utf-8") as f:
                raw = json.load(f)
            
            content = raw.get("content", "")
            links = raw.get("links", [])
            
            if not content.strip():
                st.warning(f"Empty content in {file_path.name}, skipping.")
                continue
            
            prompt = standardizer.make_standardizer_prompt(content, links)
            raw_response = await standardizer.call_azure_llm(prompt)
            
            # Log raw response
            raw_log_path = standardized_dir / f"{file_path.stem}_raw.md"
            with open(raw_log_path, "w", encoding="utf-8") as f:
                f.write(raw_response)
            
            cleaned_json = standardizer.clean_llm_response(raw_response)
            parsed_json = json.loads(cleaned_json)
            
            # Add timestamp, file source and original filename
            parsed_json["timestamp"] = datetime.now().isoformat()
            parsed_json["source_file"] = str(file_path)
            if "original_filename" in raw:
                parsed_json["original_filename"] = raw["original_filename"]
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, indent=2, ensure_ascii=False)
            
            st.session_state.standardized_files.append(output_path)
            standardized_count += 1
        except Exception as e:
            st.error(f"Error standardizing {file_path.name}: {str(e)}")
        
        # Update progress
        progress_bar.progress((i + 1) / total_files)
    
    status_text.text(f"✅ Standardized {standardized_count}/{total_files} files")
    st.session_state.standardizing_complete = True

def upload_to_mongodb():
    """Upload standardized resumes to MongoDB"""
    st.session_state.db_upload_complete = False
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize DB manager
    try:
        db_manager = ResumeDBManager()
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        return
    
    total_files = len(st.session_state.standardized_files)
    uploaded_count = 0
    
    for i, file_path in enumerate(st.session_state.standardized_files):
        status_text.text(f"Uploading {i+1}/{total_files}: {file_path.name}")
        
        try:
            with open(file_path, encoding="utf-8") as f:
                resume_data = json.load(f)
            
            # Insert or update in MongoDB
            db_manager.insert_or_update_resume(resume_data)
            uploaded_count += 1
            st.session_state.uploaded_files.append(file_path.name)
        except Exception as e:
            st.error(f"Error uploading {file_path.name}: {e}")
        
        # Update progress
        progress_bar.progress((i + 1) / total_files)
    
    status_text.text(f"✅ Uploaded {uploaded_count}/{total_files} resumes to MongoDB")
    st.session_state.db_upload_complete = True

def validate_and_reprocess_resumes(uploaded_files):
    """Validate standardized resumes and reprocess if 'name' is missing."""
    st.write("🔍 Validating standardized resumes...")
    reprocessed_count = 0

    for file_path in st.session_state.standardized_files:
        try:
            with open(file_path, encoding="utf-8") as f:
                resume_data = json.load(f)

            # Check if 'name' is missing
            if not resume_data.get("name") or len(resume_data.get("name").split()[0]) < 2:
                st.warning(f"⚠️ Missing 'name' in {file_path.name}. Reprocessing...")
                reprocessed_count += 1

                # Find the original file in uploaded files
                original_file_name = file_path.stem  # Remove .json extension
                original_file = next(
                    (file for file in uploaded_files if Path(file.name).stem == original_file_name),
                    None
                )

                if not original_file:
                    st.error(f"❌ Original file for {file_path.name} not found in uploaded files.")
                    continue

                # Save the uploaded file temporarily
                temp_file_path = temp_dir / original_file.name
                with open(temp_file_path, "wb") as f:
                    f.write(original_file.getbuffer())

                # Re-parse the original file
                parser = ResumeParserwithOCR()
                parsed_resume = parser.parse_resume(temp_file_path)

                if parsed_resume:
                    # Save parsed data
                    parsed_output_path = parsed_dir / f"{original_file_name}.json"
                    parser.save_to_json(parsed_resume, parsed_output_path)

                    # Re-standardize the parsed data
                    standardizer = ResumeStandardizer()
                    content = parsed_resume.get("content", "")
                    links = parsed_resume.get("links", [])
                    prompt = standardizer.make_standardizer_prompt(content, links)
                    raw_response = asyncio.run(standardizer.call_azure_llm(prompt))
                    cleaned_json = standardizer.clean_llm_response(raw_response)
                    parsed_json = json.loads(cleaned_json)

                    # Add metadata
                    parsed_json["timestamp"] = datetime.now().isoformat()
                    parsed_json["source_file"] = str(temp_file_path)
                    parsed_json["original_filename"] = original_file.name

                    # Save re-standardized data
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(parsed_json, f, indent=2, ensure_ascii=False)

                    st.success(f"✅ Reprocessed and standardized: {file_path.name}")
                else:
                    st.error(f"❌ Failed to re-parse {original_file.name}")
        except Exception as e:
            st.error(f"Error validating {file_path.name}: {e}")

    st.write(f"🔄 Reprocessed {reprocessed_count} resumes with missing 'name'.")

# Create temp directories for processing
temp_dir = Path(tempfile.gettempdir()) / "resume_processor"
parsed_dir = temp_dir / "parsed"
standardized_dir = temp_dir / "standardized"
for directory in [parsed_dir, standardized_dir]:
    directory.mkdir(parents=True, exist_ok=True)

# -------------------
# Page: Boolean Search Engine
# -------------------
if page == "Resume Search Engine":
    # Prevent duplicate set_page_config calls in the retriever module
    original_spc = st.set_page_config
    st.set_page_config = lambda *args, **kwargs: None

    try:
        run_retriever()
    finally:
        st.set_page_config = original_spc

# -------------------
# Page: Upload & Process Resumes
# -------------------
elif page == "Upload & Process":
    st.title("📄 Resume Processing Pipeline")
    st.markdown("""
    ### Streamlined Resume Processing
    Upload your resume files and let our AI-powered pipeline handle the rest. The system will automatically:
    1. Extract and parse content from your resumes
    2. Standardize the information into a consistent format
    3. Store the processed data in our database
    
    Supported formats: PDF, DOCX
""")

    uploaded_files = st.file_uploader(
        "📤 Upload Resume Files", 
        type=["pdf", "docx"], 
        accept_multiple_files=True,
        key="resume_uploader",
        help="Upload PDF or DOCX resume files"
    )

    # Track uploaded files in session state for persistence
    if "uploaded_file_names" not in st.session_state:
        st.session_state.uploaded_file_names = []

    # Update the list of uploaded file names
    if uploaded_files:
        current_file_names = [file.name for file in uploaded_files]
        if sorted(current_file_names) != sorted(st.session_state.uploaded_file_names):
            st.session_state.uploaded_file_names = current_file_names
            st.session_state.processing_complete = False
            st.session_state.standardizing_complete = False
            st.session_state.db_upload_complete = False
            st.session_state.processed_files = []
            st.session_state.standardized_files = []
            st.session_state.uploaded_files = []

    # Combined processing button
    if uploaded_files:
        if st.button("🚀 Process Resumes", type="primary", use_container_width=True):
            with st.spinner("Processing resumes..."):
                # Step 1: Parse
                process_uploaded_files(uploaded_files)
                st.success("✅ Parsing complete!")
                
                # Step 2: Standardize
                asyncio.run(standardize_resumes())
                st.success("✅ Standardization complete!")

                # Step 3: Validate and reprocess if necessary
                validate_and_reprocess_resumes(uploaded_files)
                
                # Step 4: Upload to MongoDB
                upload_to_mongodb()
                st.success("✅ Database upload complete!")
    else:
        st.info("👆 Please upload resume files to begin processing")

    # Display processing status
    st.subheader("📊 Processing Status")
    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        if st.session_state.processing_complete:
            st.success(f"✅ Parsed {len(st.session_state.processed_files)} files")
        else:
            st.info("⏳ Waiting for parsing...")
    with status_col2:
        if st.session_state.standardizing_complete:
            st.success(f"✅ Standardized {len(st.session_state.standardized_files)} files")
        elif st.session_state.processing_complete:
            st.info("⏳ Ready to standardize")
        else:
            st.info("⏳ Waiting for parsing...")
    with status_col3:
        if st.session_state.db_upload_complete:
            st.success(f"✅ Uploaded {len(st.session_state.uploaded_files)} files to MongoDB")
        elif st.session_state.standardizing_complete:
            st.info("⏳ Ready to upload to MongoDB")
        else:
            st.info("⏳ Waiting for standardization...")

    # Display file previews if processed
    if st.session_state.standardized_files:
        st.subheader("👀 Preview Processed Resumes")
        selected_file = st.selectbox(
            "Select a resume to preview", 
            options=[f.name for f in st.session_state.standardized_files]
        )
        if selected_file:
            file_path = standardized_dir / selected_file
            with open(file_path, "r", encoding="utf-8") as f:
                resume_data = json.load(f)
            
            # Create a more visually appealing preview
            st.markdown("---")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"### 👤 {resume_data.get('name', 'Unknown Name')}")
                st.markdown(f"📧 **Email:** {resume_data.get('email', 'No email')}")
                st.markdown(f"📱 **Phone:** {resume_data.get('phone', 'No phone')}")
                st.markdown(f"📍 **Location:** {resume_data.get('location', 'No location')}")
                if resume_data.get('skills'):
                    st.markdown("### 🛠️ Skills")
                    st.write(", ".join(resume_data.get('skills', [])))
            with col2:
                if resume_data.get('experience'):
                    st.markdown("### 💼 Experience")
                    for exp in resume_data.get('experience', [])[:2]:
                        st.markdown(f"""
                        **{exp.get('title')}** at {exp.get('company')}
                        *{exp.get('duration', 'N/A')}*
                        """)
            if st.checkbox("Show Raw JSON"):
                st.json(resume_data)

# -------------------
# Page: Database Management
# -------------------
# ...existing code...

elif page == "Database Management":
    st.title("💾 Resume Database Management")
    st.markdown("""
    ### Database Operations
    Manage and query your resume database with powerful search capabilities.
""")

    try:
        db_manager = ResumeDBManager()
        st.subheader("🔍 Query Resumes")
        query_type = st.radio("Select Query Type", ["View All Resumes", "Search by Field"])
        
        if query_type == "View All Resumes":
            if "all_resumes_results" not in st.session_state:
                st.session_state.all_resumes_results = []
            if st.button("📥 Fetch All Resumes", use_container_width=True) or st.session_state.all_resumes_results:
                with st.spinner("Fetching resumes..."):
                    if not st.session_state.all_resumes_results:
                        st.session_state.all_resumes_results = db_manager.find({})
                    results = st.session_state.all_resumes_results
                    st.success(f"Found {len(results)} resumes")
                    if results:
                        resume_data = []
                        for res in results:
                            resume_data.append({
                                "ID": res.get("_id", "N/A"),
                                "Name": res.get("name", "N/A"),
                                "Email": res.get("email", "N/A"),
                                "Skills": ", ".join(res.get("skills", [])[:3]) + ("..." if len(res.get("skills", [])) > 3 else "")
                            })
                        st.dataframe(resume_data, use_container_width=True)
                        
                        resume_options = []
                        st.session_state.resume_display_map = {}
                        for res in results:
                            display_text = f"{res.get('name', 'Unknown')} - {res.get('email', 'No email')}"
                            resume_options.append(display_text)
                            st.session_state.resume_display_map[display_text] = res
                        
                        selected_resume_option = st.selectbox(
                            "Select resume to view details", 
                            options=resume_options if resume_options else ["No resumes found"],
                            key="resume_selector"
                        )
                        
                        if selected_resume_option and "No resumes found" not in selected_resume_option:
                            selected_resume = st.session_state.resume_display_map.get(selected_resume_option)
                            if selected_resume:
                                st.json(selected_resume)

                                # Add delete button with confirmation logic
                                if "delete_confirmation" not in st.session_state:
                                    st.session_state.delete_confirmation = False

                                if not st.session_state.delete_confirmation:
                                    if st.button("🗑️ Delete Resume", key="delete_button"):
                                        st.session_state.delete_confirmation = True
                                else:
                                    # Simulate a pop-up-like experience
                                    with st.container():
                                        st.error("⚠️ Are you sure you want to delete this resume? This action cannot be undone.")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            if st.button("Yes, Delete", key="confirm_delete_button"):
                                                try:
                                                    db_manager.delete_resume({"_id": selected_resume["_id"]})
                                                    st.success(f"✅ Deleted resume: {selected_resume.get('name', 'Unknown')}")
                                                    # Refresh the results after deletion
                                                    st.session_state.all_resumes_results = db_manager.find({})
                                                    st.session_state.delete_confirmation = False  # Reset confirmation state
                                                except Exception as e:
                                                    st.error(f"Error deleting resume: {e}")
                                        with col2:
                                            if st.button("Cancel", key="cancel_delete_button"):
                                                st.info("Deletion canceled.")
                                                st.session_state.delete_confirmation = False  # Reset confirmation state
                            else:
                                st.error("Could not find the selected resume. Please try again.")
        
        elif query_type == "Search by Field":
            col1, col2 = st.columns(2)
            with col1:
                search_field = st.selectbox(
                "Search Field", 
                ["name", "email", "skills", "experience.company", "education.institution"]
            )
            with col2:
                search_value = st.text_input("Search Value")
            
            if st.button("🔍 Search", use_container_width=True):
                if search_value:
                    query = {}
                    if search_field == "skills":
                        query = {search_field: {"$in": [search_value]}}
                    elif "." in search_field:
                        query = {search_field: {"$regex": search_value, "$options": "i"}}
                    else:
                        query = {search_field: {"$regex": search_value, "$options": "i"}}
                    
                    with st.spinner("Searching..."):
                        results = db_manager.find(query)
                        if results:
                            st.success(f"Found {len(results)} matching resumes")
                            search_options = []
                            search_map = {}
                            for res in results:
                                display_text = f"{res.get('name', 'Unknown')} - {res.get('email', 'No email')}"
                                search_options.append(display_text)
                                search_map[display_text] = res
                            
                            # Store search results and map in session state
                            st.session_state.search_options = search_options
                            st.session_state.search_map = search_map
                        else:
                            st.warning("No matching resumes found")
                else:
                    st.warning("Please enter a search value")
            
            # Display search results if available
            if "search_options" in st.session_state and st.session_state.search_options:
                selected_search_result = st.selectbox(
                    "Select resume to view details", 
                    options=st.session_state.search_options,
                    key="search_selector"
                )
                
                if selected_search_result:
                    selected_resume = st.session_state.search_map.get(selected_search_result)
                    if selected_resume:
                        st.json(selected_resume)

                        # Add delete button with confirmation logic
                        if "delete_confirmation" not in st.session_state:
                            st.session_state.delete_confirmation = False

                        if not st.session_state.delete_confirmation:
                            if st.button("🗑️ Delete Resume", key="delete_button"):
                                st.session_state.delete_confirmation = True
                        else:
                            # Simulate a pop-up-like experience
                            with st.container():
                                st.error("⚠️ Are you sure you want to delete this resume? This action cannot be undone.")
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button("Yes, Delete", key="confirm_delete_button"):
                                        try:
                                            db_manager.delete_resume({"_id": selected_resume["_id"]})
                                            st.success(f"✅ Deleted resume: {selected_resume.get('name', 'Unknown')}")
                                            # Refresh the results after deletion
                                            st.session_state.all_resumes_results = db_manager.find({})
                                            st.session_state.delete_confirmation = False  # Reset confirmation state
                                        except Exception as e:
                                            st.error(f"Error deleting resume: {e}")
                                with col2:
                                    if st.button("Cancel", key="cancel_delete_button"):
                                        st.info("Deletion canceled.")
                                        st.session_state.delete_confirmation = False  # Reset confirmation state
                    else:
                        st.error("Could not find the selected resume. Please try again.")
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
# # -------------------
# # Page: Settings
# # -------------------
# elif page == "Settings":
#     st.title("Settings")
#     st.subheader("API Keys")
#     llama_key = st.text_input("LLAMA_CLOUD_API_KEY", value=st.secrets["azure_openai"]["api_key"], type="password")
#     azure_key = st.text_input("AZURE_OPENAI_API_KEY", value=st.secrets["azure_openai"]["endpoint"], type="password")
#     azure_endpoint = st.text_input("AZURE_OPENAI_ENDPOINT", value=st.secrets["azure_openai"]["deployment"])# noqa
#     azure_deployment = st.text_input("AZURE_OPENAI_DEPLOYMENT", value=st.secrets["azure_openai"].get("api_version", "2024-08-01-preview"))

#     st.subheader("MongoDB Settings")
#     mongo_uri = st.text_input("MongoDB URI", value=st.secrets["mongo"]["uri"], type="password")
#     db_name = st.text_input("Database Name", value=st.secrets["mongo"]["db_name"])
#     collection_name = st.text_input("Collection Name", value=st.secrets["mongo"]["collection_name"])

#     if st.button("Save Settings"):
#         secrets_content = f"""
# [LLAMA_CLOUD_API_KEY] = "{llama_key}"
# [AZURE_OPENAI_API_KEY] = "{azure_key}"
# [AZURE_OPENAI_ENDPOINT] = "{azure_endpoint}"
# [AZURE_OPENAI_DEPLOYMENT] = "{azure_deployment}"
# [MONGO_URI] = "{mongo_uri}"
# [DB_NAME] = "{db_name}"
# [COLLECTION_NAME] = "{collection_name}"
# """
#         secrets_path = Path(".streamlit/secrets.toml")
#         secrets_path.parent.mkdir(parents=True, exist_ok=True)
#         secrets_path.write_text(secrets_content)
#         st.success("Settings saved to secrets.toml!")

#     st.subheader("Create config.py")
#     if st.button("Generate config.py"):
#         config_content = f"""# config.py
# # MongoDB settings
# MONGO_URI = "{mongo_uri}"
# DB_NAME = "{db_name}"
# COLLECTION_NAME = "{collection_name}"
# """
#         with open("config.py", "w") as f:
#             f.write(config_content)
#         st.success("config.py created!")
#         st.code(config_content, language="python")

