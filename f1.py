import openai
import re
# ------------------- Azure OpenAI Configuration -------------------


AZURE_OPENAI_API_KEY = "3xPDwbMLxnRrhsGke6eymlGOr4poo5h3jmHH5jZgZRzFuw1kwLXTJQQJ99AJACYeBjFXJ3w3AAABACOGiocA"
AZURE_OPENAI_ENDPOINT = "https://us-tax-law-rag-demo.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"
AZURE_OPENAI_API_VERSION = "2024-08-01-preview"  # API version you're using

# Create a reusable OpenAI client instance (Azure)
openai_client = openai.AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)
def convert_natural_language_to_boolean(nl_query):
    prompt = f"""Convert the following natural language query into a Boolean search query.
 use logical operators like AND, OR, and NOT in uppercase.

Example 1:
Input: Show me candidates skilled in Java and Spring Boot
Output: java AND springboot

Example 2:
Input: Find someone who knows Python or data science
Output: python OR datascience

Example 3:
Input: I want profiles with either machine learning or deep learning but not statistics
Output: (machinelearning OR deeplearning) AND NOT statistics

Now, convert this:
Input: {nl_query}
Output:"""

    response = openai_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()
def convert_nl_query(query):
    return convert_natural_language_to_boolean(query)

def preprocess_query(query):
    query = convert_nl_query(query)

    def replace_quoted(match):
        quoted_phrases = {}
        phrase = match.group(1)
        placeholder = f"QUOTED_PHRASE_{placeholder_counter}"
        quoted_phrases[placeholder] = phrase
        placeholder_counter += 1
        return placeholder

    processed_query = re.sub(r'"([^"]+)"', replace_quoted, query)
    return processed_query
query = str(input("Enter your query: "))
ans = convert_nl_query(query)
print(ans)