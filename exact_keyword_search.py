import re
import openai
import streamlit as st
from pymongo import MongoClient
from typing import List, Set, Dict, Union
# ------------------- Azure OpenAI Configuration -------------------

# Load Azure OpenAI secrets from .streamlit/secrets.toml
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
    prompt = f"""Convert the following natural language query into a Boolean search query and remember number of operators is 1 less than number of keywords.
        also if keyword is of multiple words, then give whole keyword in double quotes.
        Example 1:
        Input: Show me candidates skilled in Java and Spring Boot
        Output: java and "spring boot"

        Example 2:
        Input: Find someone who knows Python or data science
        Output: python or "data science"

        Example 3:
        Input: I want profiles with either machine learning or deep learning but not statistics
        Output: ("machine learning" or "deep learning") and not statistics

        Now, convert this:
        Input: {nl_query}
        Output:
        do not show ###output:### in output"""

    response = openai_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def connect_to_mongo(uri="mongodb+srv://utkarshsingh:G8stmYP7mbRXp3zR@clusterforresumes.trfczld.mongodb.net/", db_name="resume_db", collection_name="resumes"):
    client = MongoClient(uri)
    db = client[db_name]
    return db[collection_name]

def flatten_document(d, parent_key='', sep='_'):
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_document(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, sub_item in enumerate(v):
                    items.extend(flatten_document(sub_item, f"{new_key}_{i}", sep=sep).items())
            else:
                items.append((new_key, v))
    elif isinstance(d, list):
        for i, sub_item in enumerate(d):
            items.extend(flatten_document(sub_item, f"{parent_key}_{i}", sep=sep).items())
    else:
        items.append((parent_key, d))
    return dict(items)

def search_keyword(collection, keyword: str) -> Set[str]:
    matching_names = set()
    
    for doc in collection.find():
        name = doc.get('name', '')
        if not name:
            continue
            
        flattened_doc = flatten_document(doc)
        for key, value in flattened_doc.items():
            if isinstance(value, str) and keyword in value.lower():
                matching_names.add(name)
                break
                
    return matching_names

# --------------------- Boolean Expression Parsing ---------------------

def tokenize(expression: str) -> List[str]:
    # Match quoted phrases as a single token, along with AND, OR, (, ) and words
    pattern = r'"[^"]+"|\(|\)|\bAND\b|\bOR\b|\S+'
    tokens = re.findall(pattern, expression, flags=re.IGNORECASE)
    
    # Normalize tokens (remove quotes from phrases and lowercase keywords)
    normalized_tokens = []
    for token in tokens:
        upper = token.upper()
        if upper in ('AND', 'OR', '(', ')'):
            normalized_tokens.append(upper)
        else:
            # Remove quotes if present
            token = token.strip('"').lower()
            normalized_tokens.append(token)
    
    return normalized_tokens


def to_postfix(tokens: List[str]) -> List[str]:
    """Convert infix Boolean expression to postfix (RPN) using Shunting Yard algorithm."""
    precedence = {'AND': 2, 'OR': 1}
    output = []
    stack = []
    
    for token in tokens:
        upper = token.upper()
        if upper in ('AND', 'OR'):
            while stack and stack[-1] != '(' and precedence.get(stack[-1], 0) >= precedence[upper]:
                output.append(stack.pop())
            stack.append(upper)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()  # Pop the '('
        else:
            output.append(token.lower())  # Operand (keyword)
    
    while stack:
        output.append(stack.pop())
    
    return output

def evaluate_postfix(postfix_tokens: List[str], collection) -> Set[str]:
    stack: List[Union[str, Set[str]]] = []
    cache: Dict[str, Set[str]] = {}
    
    for token in postfix_tokens:
        if token.upper() == 'AND':
            right = stack.pop()
            left = stack.pop()
            stack.append(left & right)
        elif token.upper() == 'OR':
            right = stack.pop()
            left = stack.pop()
            stack.append(left | right)
        else:
            if token not in cache:
                cache[token] = search_keyword(collection, token)
            stack.append(cache[token])
    
    return stack[0] if stack else set()

# --------------------- Main Logic ---------------------

def run_search(query: str):
    print(f"🔍 Parsing query: {query}")
    tokens = tokenize(query)
    print(f"🧩 Tokens: {tokens}")
    postfix = to_postfix(tokens)
    print(f"📥 Postfix (RPN): {postfix}")
    
    collection = connect_to_mongo()
    result = evaluate_postfix(postfix, collection)

    print("\n📝 Final matching resumes:")
    print("------------------------------------------------------")
    for name in sorted(result):
        print(f"   - {name}")
    print(f"\n📊 Total matching resumes: {len(result)}")
    print("\n🏁 Search complete.")

# Example usage:
if __name__ == "__main__":
    query = str(input("Enter your query: "))
    query_string = convert_natural_language_to_boolean(query)
    print(f"Converted query: {query_string}")
    run_search(query_string)
