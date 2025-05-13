import json
import re
import streamlit as st
from pymongo import MongoClient
from boolean.boolean import BooleanAlgebra, Symbol, AND, OR
import config
import openai

# ------------------- Azure OpenAI Configuration -------------------

AZURE_OPENAI_API_KEY = st.secrets["azure_openai"]["api_key"]
AZURE_OPENAI_ENDPOINT = st.secrets["azure_openai"]["endpoint"]
AZURE_OPENAI_DEPLOYMENT = st.secrets["azure_openai"]["deployment"]
AZURE_OPENAI_API_VERSION = st.secrets["azure_openai"]["api_version"]

# Create a reusable OpenAI client instance (Azure)
openai_client = openai.AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

def convert_natural_language_to_boolean(nl_query):
    prompt = f"""Convert the following natural language query into a Boolean search query.
        Example 1:
        Input: Show me candidates skilled in Java and Spring Boot
        Output: java and springboot

        Example 2:
        Input: Find someone who knows Python or data science
        Output: python or datascience

        Example 3:
        Input: I want profiles with either machine learning or deep learning but not statistics
        Output: (machinelearning or deeplearning) and not statistics

        Now, convert this:
        Input: {nl_query}
        Output:"""

    response = openai_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# Boolean parser class
class BooleanSearchParser:
    def __init__(self):
        self.algebra = BooleanAlgebra()
        self.quoted_phrases = {}
        self.placeholder_counter = 0

    def convert_nl_query(self, query):
        return convert_natural_language_to_boolean(query)

    def preprocess_query(self, query):
        query = self.convert_nl_query(query)

        def replace_quoted(match):
            phrase = match.group(1)
            placeholder = f"QUOTED_PHRASE_{self.placeholder_counter}"
            self.quoted_phrases[placeholder] = phrase
            self.placeholder_counter += 1
            return placeholder

        processed_query = re.sub(r'"([^"]+)"', replace_quoted, query)
        return processed_query

    def parse_query(self, query):
        try:
            self.quoted_phrases = {}
            self.placeholder_counter = 0
            # First extract quoted phrases
            processed_query = self.preprocess_query(query)
            # Then parse the Boolean expression
            return self.algebra.parse(processed_query)
        except Exception as e:
            raise ValueError(f"Invalid Boolean query: {e}")

def normalize(text: str) -> str:
    """Lowercase, split CamelCase, remove noise, then inject merged bigrams & halves."""
    # 1) Split CamelCase: "HuggingFace" → "Hugging Face"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # 2) Lowercase & basic cleanup
    text = text.lower()
    text = re.sub(r'(?<![\w@])\.net(?![\w.])', ' dotnet ', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 3) Capture quoted phrases in the _source text_ and append the no-space form
    quoted_phrases = re.findall(r'"([^"]+)"', text)
    for phrase in quoted_phrases:
        text += " " + phrase.replace(" ", "")

    # 4) Strip quotation marks, remove symbols
    text = text.replace('"', '')
    text = re.sub(r'[^\w\s]', ' ', text)

    # 5) Tokenize & inject bigrams + halves
    words = text.split()
    # 5a) adjacent-word bigrams
    for i in range(len(words) - 1):
        text += " " + words[i] + words[i+1]
    # 5b) for long merged tokens, also inject a halved split
    for tok in words:
        if len(tok) > 8:
            mid = len(tok) // 2
            text += f" {tok[:mid]} {tok[mid:]}"

    # 6) Normalize whitespace
    return re.sub(r'\s+', ' ', text).strip()

def flatten_json(obj) -> str:
    parts = []
    def recurse(x):
        if isinstance(x, str):
            parts.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                recurse(v)
        elif isinstance(x, list):
            for i in x:
                recurse(i)
    recurse(obj)
    return " ".join(parts)

def evaluate_expression(expr, text, quoted_phrases=None):
    """Recursively evaluate Boolean expression against text, with substring fallback."""
    quoted_phrases = quoted_phrases or {}
    
    if isinstance(expr, Symbol):
        term = str(expr.obj).lower()
        
        # 1) exact-phrase placeholders
        if term.startswith("QUOTED_PHRASE_") and term in quoted_phrases:
            phrase = quoted_phrases[term].lower()
            return phrase in text
        
        # 2) word-boundary match
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text):
            return True
        
        # 3) fallback: substring
        return term in text

    elif isinstance(expr, AND):
        return all(evaluate_expression(arg, text, quoted_phrases) for arg in expr.args)
    elif isinstance(expr, OR):
        return any(evaluate_expression(arg, text, quoted_phrases) for arg in expr.args)
    
    return False

def display_json(data):
    if "_id" in data:
        del data["_id"]
    
    st.json(data)

def normalize_boolean_operators(query):
    # Replace lowercase boolean operators with uppercase versions (whole words only)
    query = re.sub(r'\band\b', 'AND', query, flags=re.IGNORECASE)
    query = re.sub(r'\bor\b', 'OR', query, flags=re.IGNORECASE)
    query = re.sub(r'\bnot\b', 'NOT', query, flags=re.IGNORECASE)
    return query

def search_based_on_keyword_length(query, docs):
    """
    Search based on keyword length:
    - If keyword length > 4: Use boolean search
    - If keyword length <= 4: Use Linux-based word search
    """
    # Split query into individual keywords
    keywords = query.lower().split()
    
    # Check if any keyword is longer than 4 characters
    use_boolean = any(len(keyword) > 4 for keyword in keywords)
    
    if use_boolean:
        # Use existing boolean search logic
        bsp = BooleanSearchParser()
        try:
            parsed_query = bsp.parse_query(query)
            matching_docs = []
            for doc in docs:
                raw_text = flatten_json(doc)
                norm_text = normalize(raw_text)
                if evaluate_expression(parsed_query, norm_text, bsp.quoted_phrases):
                    matching_docs.append(doc)
            return matching_docs
        except Exception as e:
            st.error(f"❌ Error in boolean search: {e}")
            return []
    else:
        # Use Linux-based word search
        matching_docs = []
        for doc in docs:
            raw_text = flatten_json(doc).lower()
            # Check if all keywords are present in the text
            if all(keyword in raw_text for keyword in keywords):
                matching_docs.append(doc)
        return matching_docs

def main():
    st.set_page_config(
        page_title="HR Bot Resume Search", 
        page_icon="📄", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for improved UI
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        width: 100%;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        border-left: 5px solid #0068c9;
    }
    .card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .candidate-name {
        font-size: 20px;
        font-weight: bold;
        color: #0068c9;
    }
    .contact-info {
        color: #444;
        margin: 10px 0;
    }
    .result-count {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar for search controls
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/find-matching-job.png", width=80)
        st.title("HR Bot Resume Search")
        
        st.markdown("### Search Filters")
        search_query = st.text_input("🧠 Enter your search query:", placeholder="e.g., Python AND MachineLearning")
        search_query = convert_natural_language_to_boolean(search_query)
        if search_query:
            search_query = normalize_boolean_operators(search_query)
        
        with st.expander("ℹ️ How to Search ??"):
            st.markdown("""
            ### 🔍 Boolean Search Tips
            - **Simple keyword**: Type keywords directly like `Python`
            - **AND operator**: Match multiple skills: `Python AND Django`
            - **OR operator**: Match alternatives: `JavaScript OR TypeScript`
            - **Grouped logic**: Combine filters using parentheses:  
            e.g., `(Python OR Java) AND (AWS OR Azure)`
            - **Multi-word skills** (like 2+ word technologies):
                - Write them without spaces: `MachineLearning`, `HuggingFace`
                - OR use Boolean grouping: `(Machine AND Learning)`, `(Hugging AND Face)`
                - ✅ All the following are treated the same:
                - `MachineLearning`
                - `machinelearning`
                - `(Machine AND Learning)`
                - `HuggingFace`, `huggingface`, `(Hugging AND Face)`

            💡 _Avoid spaces between multi-word skills unless using Boolean logic explicitly._
            """)

    # Main content
    st.title("🔎 Looking for some candidates?")
    
    if not search_query:
        st.info("👈 Enter a search query in the sidebar to begin searching.")
        return

    # Connect to MongoDB
    try:
        with st.spinner("Connecting to database..."):
            client = MongoClient(config.MONGO_URI)
            coll = client[config.DB_NAME][config.COLLECTION_NAME]
            docs = list(coll.find({}))
            st.success(f"📁 Loaded {len(docs)} resumes from database")
    except Exception as e:
        st.error(f"❌ Failed to load resumes: {e}")
        return

    # Search resumes
    st.subheader("🔍 Searching resumes...")
    progress_bar = st.progress(0)
    
    # Store full documents for matched candidates
    matching_docs = search_based_on_keyword_length(search_query, docs)
    
    progress_bar.empty()

    # Display results
    if matching_docs:
        st.markdown(f"<div class='result-count'>✅ Found {len(matching_docs)} matching candidates</div>", unsafe_allow_html=True)
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Card View", "Table View"])
        
        with tab1:
            # Card view
            for doc in matching_docs:
                with st.container():
                    st.markdown(f"""
                    <div class="card">
                        <div class="candidate-name">{doc.get('name', 'Unknown Candidate')}</div>
                        <div class="contact-info">
                            📧 {doc.get('email', 'No email provided')} | 
                            📱 {doc.get('phone', 'No phone provided')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([4, 1])
                    
                    # Extract top skills/keywords for preview
                    skills = doc.get('skills', [])
                    if skills:
                        if isinstance(skills, list):
                            skill_text = ", ".join(skills[:5])
                            if len(skills) > 5:
                                skill_text += "..."
                        else:
                            skill_text = str(skills)
                        col1.markdown(f"**Skills**: {skill_text}")
                    
                    # View details button
                    if col2.button("View Details", key=f"view_{doc.get('_id', idx)}"):
                        st.session_state[f"show_details_{doc.get('_id', idx)}"] = True
                    
                    # Show details if button was clicked
                    if st.session_state.get(f"show_details_{doc.get('_id', idx)}", False):
                        with st.expander("📄 Full Resume Details", expanded=True):
                            tabs = st.tabs(["Formatted View", "JSON View"])
                            
                            with tabs[0]:
                                # Formatted structured view
                                st.subheader(f"{doc.get('name', 'Candidate')} - Profile")
                                
                                # Basic information
                                st.markdown("### 👤 Basic Information")
                                col1, col2 = st.columns(2)
                                col1.markdown(f"**Name:** {doc.get('name', 'N/A')}")
                                col1.markdown(f"**Email:** {doc.get('email', 'N/A')}")
                                col2.markdown(f"**Phone:** {doc.get('phone', 'N/A')}")
                                col2.markdown(f"**Location:** {doc.get('location', 'N/A')}")
                                
                                # Education
                                if 'education' in doc:
                                    st.markdown("### 🎓 Education")
                                    if isinstance(doc['education'], list):
                                        for edu in doc['education']:
                                            if isinstance(edu, dict):
                                                st.markdown(f"**{edu.get('degree', 'Degree')}** - {edu.get('institution', 'Institution')}")
                                                st.markdown(f"{edu.get('start_date', '')} - {edu.get('end_date', '')} | {edu.get('location', '')}")
                                            else:
                                                st.markdown(f"- {edu}")
                                    else:
                                        st.markdown(f"- {doc['education']}")
                                
                                # Experience
                                if 'experience' in doc:
                                    st.markdown("### 💼 Experience")
                                    if isinstance(doc['experience'], list):
                                        for exp in doc['experience']:
                                            if isinstance(exp, dict):
                                                st.markdown(f"**{exp.get('title', 'Role')}** at {exp.get('company', 'Company')}")
                                                st.markdown(f"{exp.get('start_date', '')} - {exp.get('end_date', '')} | {exp.get('location', '')}")
                                                st.markdown(f"{exp.get('description', '')}")
                                            else:
                                                st.markdown(f"- {exp}")
                                    else:
                                        st.markdown(f"- {doc['experience']}")

                                # Projects
                                if 'projects' in doc:
                                    st.markdown("### 🛠️ Projects")
                                    if isinstance(doc['projects'], list):
                                        for proj in doc['projects']:
                                            if isinstance(proj, dict):
                                                st.markdown(f"**{proj.get('title', 'Project')}**")
                                                st.markdown(f"{proj.get('description', '')}")
                                            else:
                                                st.markdown(f"- {proj}")
                                    else:
                                        st.markdown(f"- {doc['projects']}")

                                
                                # Skills
                                if 'skills' in doc:
                                    st.markdown("### 🛠️ Skills")
                                    if isinstance(doc['skills'], list):
                                        st.markdown(", ".join(doc['skills']))
                                    else:
                                        st.markdown(doc['skills'])
                                
                                

                                # Certifications
                                if 'certifications' in doc:
                                    st.markdown("### 📜 Certifications")
                                    if isinstance(doc['certifications'], list):
                                        for cert in doc['certifications']:
                                            if isinstance(cert, dict):
                                                st.markdown(f"**{cert.get('title', 'Certification')}** - {cert.get('issuer', '')} ({cert.get('year', '')})")
                                                if 'link' in cert:
                                                    st.markdown(f"[🔗 View Certificate]({cert['link']})")
                                            else:
                                                st.markdown(f"- {cert}")
                                    else:
                                        st.markdown(f"- {doc['certifications']}")

                                # Languages
                                if 'languages' in doc and doc['languages']:
                                    st.markdown("### 🌍 Languages")
                                    if isinstance(doc['languages'], list):
                                        st.markdown(", ".join(doc['languages']))
                                    else:
                                        st.markdown(doc['languages'])

                                # Social Profiles
                                if 'social_profiles' in doc:
                                    st.markdown("### 🌐 Social Profiles")
                                    if isinstance(doc['social_profiles'], list):
                                        for profile in doc['social_profiles']:
                                            if isinstance(profile, dict):
                                                st.markdown(f"[{profile.get('platform', 'Profile')}]({profile.get('link', '#')})")
                                            else:
                                                st.markdown(f"- {profile}")
                                    else:
                                        st.markdown(f"- {doc['social_profiles']}")

                            
                            with tabs[1]:
                                # Raw JSON view with pretty formatting
                                display_json(doc)
        
        with tab2:
            # Table view for comparison
            table_data = []
            for doc in matching_docs:
                row = {
                    "Name": doc.get('name', 'Unknown'),
                    "Email": doc.get('email', 'N/A'),
                    "Phone": doc.get('phone', 'N/A'),
                    "Location": doc.get('location', 'N/A'),
                }
                
                # Add skills as comma-separated string
                skills = doc.get('skills', [])
                if isinstance(skills, list):
                    row["Skills"] = ", ".join(skills[:3]) + ("..." if len(skills) > 3 else "")
                else:
                    row["Skills"] = str(skills)
                
                table_data.append(row)
            
            st.dataframe(table_data, use_container_width=True)
    else:
        st.info("🔎 No resumes matched your search query. Try adjusting your terms.")
        st.markdown("""
        **Tips to improve results:**
        - Use broader terms
        - Try using OR instead of AND
        - Check for typos in your search query
        - Simplify complex Boolean expressions
        """)

def run_retriever():
    if 'init' not in st.session_state:
        st.session_state.init = True
        # Initialize any other session state variables here
    
    main()
