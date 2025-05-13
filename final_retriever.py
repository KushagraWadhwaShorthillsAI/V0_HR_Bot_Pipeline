import json
import re
import streamlit as st
from pymongo import MongoClient
from boolean.boolean import BooleanAlgebra, Symbol, AND, OR
import config

# Boolean parser class
class BooleanSearchParser:
    def __init__(self):
        self.algebra = BooleanAlgebra()
        self.quoted_phrases = {}
        self.placeholder_counter = 0

    def preprocess_query(self, query):
        """Extract quoted phrases and replace them with placeholders"""
        def replace_quoted(match):
            phrase = match.group(1)
            placeholder = f"QUOTED_PHRASE_{self.placeholder_counter}"
            self.quoted_phrases[placeholder] = phrase
            self.placeholder_counter += 1
            return placeholder
        
        # Replace quoted phrases with placeholders
        processed_query = re.sub(r'"([^"]+)"', replace_quoted, query)
        return processed_query

    def parse_query(self, query):
        try:
            self.quoted_phrases   = {}
            self.placeholder_counter = 0
            # First extract quoted phrases
            processed_query = self.preprocess_query(query)
            # Then parse the Boolean expression
            return self.algebra.parse(processed_query)
        except Exception as e:
            raise ValueError(f"Invalid Boolean query: {e}")

# Normalizer

def normalize(text: str) -> str:
    """Lowercase, split CamelCase, remove noise, then inject merged bigrams & halves."""
    # 1) Split CamelCase: "HuggingFace" → "Hugging Face"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # 2) Lowercase & basic cleanup
    text = text.lower()
    text = re.sub(r'(?<![\w@])\.net(?![\w.])', ' dotnet ', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 3) Handle multi-word keywords by preserving spaces
    # First, store the original text with spaces
    original_text = text

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

    # 6) Normalize whitespace and ensure single spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 7) Remove duplicate words
    words = text.split()
    unique_words = []
    seen_words = set()
    for word in words:
        if word not in seen_words:
            seen_words.add(word)
            unique_words.append(word)
    
    # 8) Add back the original text with spaces to preserve multi-word matches
    text = ' '.join(unique_words) + " " + original_text
    
    return text

def process_natural_language_query(query: str) -> str:
    """Process natural language query to extract keywords and boolean logic."""
    # Common boolean indicators
    boolean_indicators = {
        'and': 'AND',
        'or': 'OR',
        'not': 'NOT',
        'with': 'AND',
        'without': 'NOT',
        'plus': 'AND',
        'minus': 'NOT',
        'either': 'OR',
        'both': 'AND',
        'neither': 'NOT',
        'but': 'AND',
        'except': 'NOT'
    }
    
    # Convert to lowercase for processing
    query = query.lower()
    
    # Replace boolean indicators with their operators
    for indicator, operator in boolean_indicators.items():
        # Use word boundaries to ensure we're matching whole words
        query = re.sub(r'\b' + indicator + r'\b', operator, query)
    
    # Handle parentheses for grouping
    # Add parentheses around phrases that should be grouped
    query = re.sub(r'([^()]+)(?:\s+(?:and|or|not)\s+[^()]+)+', r'(\1)', query)
    
    # Clean up extra spaces
    query = re.sub(r'\s+', ' ', query).strip()
    
    return query

# Flattener
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

# Evaluator

def evaluate_expression(expr, text, quoted_phrases=None):
    """Recursively evaluate Boolean expression against text, with substring fallback."""
    quoted_phrases = quoted_phrases or {}
    
    if isinstance(expr, Symbol):
        term = str(expr.obj).lower()
        
        # 1) exact‐phrase placeholders
        if term.startswith("QUOTED_PHRASE_") and term in quoted_phrases:
            phrase = quoted_phrases[term].lower()
            return phrase in text
        
        # 2) Special handling for short terms (4 or fewer characters)
        if len(term) <= 4:
            # For short terms, require exact word boundary match
            pattern = r'\b' + re.escape(term) + r'\b'
            return bool(re.search(pattern, text))
        
        # 3) For longer terms, use word boundary match
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text):
            return True
        
        # 4) fallback: substring (only for terms longer than 4 characters)
        return term in text

    elif isinstance(expr, AND):
        # For AND operations, ensure all terms are found in different positions
        found_positions = set()
        for arg in expr.args:
            if isinstance(arg, Symbol):
                term = str(arg.obj).lower()
                if term.startswith("QUOTED_PHRASE_") and term in quoted_phrases:
                    phrase = quoted_phrases[term].lower()
                    pos = text.find(phrase)
                    if pos == -1:
                        return False
                    found_positions.add(pos)
                else:
                    pattern = r'\b' + re.escape(term) + r'\b'
                    match = re.search(pattern, text)
                    if not match:
                        return False
                    found_positions.add(match.start())
            else:
                if not evaluate_expression(arg, text, quoted_phrases):
                    return False
        return True

    elif isinstance(expr, OR):
        # For OR operations, check if any term matches
        for arg in expr.args:
            if evaluate_expression(arg, text, quoted_phrases):
                return True
        return False
    
    return False

def display_json(data):
    if "_id" in data:
        del data["_id"]
    
    st.json(data)

# Main Streamlit App
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
        search_query = st.text_input("🧠 Enter your search query:", placeholder="e.g., Python AND Machine Learning")
        
        with st.expander("ℹ️ How to Search ??"):
            st.markdown("""
            ### 🔍 Search Tips
            - **Natural Language**: Type queries like "Find candidates with Python and Machine Learning experience"
            - **Simple keyword**: Type keywords directly like `Python`
            - **AND operator**: Match multiple skills: `Python AND Django`
            - **OR operator**: Match alternatives: `JavaScript OR TypeScript`
            - **Grouped logic**: Combine filters using parentheses:  
            e.g., `(Python OR Java) AND (AWS OR Azure)`
            - **Multi-word skills**: Write them naturally with spaces:
                - `Machine Learning`
                - `Artificial Intelligence`
                - `Data Science`
            - **Boolean operators**: Use AND, OR, NOT to combine terms
            - **Natural language**: The system understands phrases like:
                - "Find candidates with Python and Machine Learning"
                - "Show me developers who know either Java or Python"
                - "Candidates with AWS but not Azure"
            """)

        # Process the query if it's not empty
        if search_query:
            # First try natural language processing
            processed_query = process_natural_language_query(search_query)
            
            # If the query contains boolean operators, use boolean search
            if 'AND' in processed_query or 'OR' in processed_query or 'NOT' in processed_query:
                try:
                    bsp = BooleanSearchParser()
                    parsed_query = bsp.parse_query(processed_query)
                except Exception as e:
                    st.error(f"❌ Error parsing query: {e}")
                    return
            else:
                # For simple keyword searches, just use the processed query
                parsed_query = Symbol(processed_query.lower())

    # Main content
    st.title("🔎 Looking for some candidates?")
    
    if not search_query:
        st.info("👈 Enter a search query in the sidebar to begin searching.")
        
        # Sample placeholders when no search is performed
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🚀 Features
            - **Boolean Logic**: Complex search queries
            - **Fast Search**: Optimized algorithm
            - **Detailed View**: See complete candidate profiles
            - **User-friendly**: Intuitive interface
            """)
        with col2:
            st.markdown("""
            ### 💡 Example Queries
            - `Python AND (Django OR Flask)`
            - `JavaScript AND React`
            - `"Machine Learning" AND (Python OR R)`
            - `AWS OR Azure`
            """)
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
    
    # Store unique documents using a dictionary with _id as key
    unique_matching_docs = {}
    
    for idx, doc in enumerate(docs):
        try:
            doc_id = str(doc.get('_id'))
            
            # Skip if we've already processed this document
            if doc_id in unique_matching_docs:
                continue
                
            raw_text = flatten_json(doc)
            norm_text = normalize(raw_text)
            
            # Debug output for search terms
            if st.session_state.get('debug_search', False):
                st.write(f"Searching in document {doc_id}:")
                st.write(f"Normalized text: {norm_text[:200]}...")
            
            if evaluate_expression(parsed_query, norm_text, bsp.quoted_phrases):
                unique_matching_docs[doc_id] = doc
        except Exception as e:
            st.warning(f"⚠️ Error processing document {doc.get('_id')}: {e}")
        progress_bar.progress((idx + 1) / len(docs))

    progress_bar.empty()

    # Get list of unique matching documents
    matching_docs_list = list(unique_matching_docs.values())

    # Display results
    if matching_docs_list:
        st.markdown(f"<div class='result-count'>✅ Found {len(matching_docs_list)} matching candidates</div>", unsafe_allow_html=True)
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Card View", "Table View"])
        
        with tab1:
            # Card view
            for doc in matching_docs_list:
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
                    
                    # View details button with unique key
                    button_key = f"view_{doc.get('_id')}"
                    if col2.button("View Details", key=button_key):
                        st.session_state[f"show_details_{doc.get('_id')}"] = True
                    
                    # Show details if button was clicked
                    if st.session_state.get(f"show_details_{doc.get('_id')}", False):
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
            for doc in matching_docs_list:
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
