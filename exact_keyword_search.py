import sys
from pymongo import MongoClient

def connect_to_mongo(uri="mongodb+srv://utkarshsingh:G8stmYP7mbRXp3zR@clusterforresumes.trfczld.mongodb.net/", db_name="resume_db", collection_name="resumes"):
    client = MongoClient(uri)
    db = client[db_name]
    return db[collection_name]

# Function to flatten the document recursively
def flatten_document(d, parent_key='', sep='_'):
    items = []
    if isinstance(d, dict):  # Ensure we are only dealing with dictionaries
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_document(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, sub_item in enumerate(v):
                    items.extend(flatten_document(sub_item, f"{new_key}_{i}", sep=sep).items())
            else:
                items.append((new_key, v))
    elif isinstance(d, list):  # Handle lists by treating each item as a dictionary
        for i, sub_item in enumerate(d):
            items.extend(flatten_document(sub_item, f"{parent_key}_{i}", sep=sep).items())
    else:
        items.append((parent_key, d))  # Directly append the value if it's not a dict or list
    return dict(items)

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <keyword1> [<keyword2> ...]")
        sys.exit(1)

    keywords = [kw.lower() for kw in sys.argv[1:]]
    total_keywords = len(keywords)
    seen_names = set()

    print(f"🔍 Searching for keywords: {keywords} in MongoDB resumes collection...")

    collection = connect_to_mongo()

    for required_matches in range(total_keywords, 0, -1):
        print(f"\n🔎 Resumes matching exactly {required_matches} keyword(s):")
        print("------------------------------------------------------")
        found_any = False

        # Fetch all resumes
        for doc in collection.find():
            name = doc.get('name', '')  # Use 'name' instead of '_id'
            if not name or name in seen_names:
                continue

            # Flatten the document
            flattened_doc = flatten_document(doc)
            matched_keywords = []

            # Search through all flattened key-value pairs
            for key, value in flattened_doc.items():
                if isinstance(value, str):
                    content_lower = value.lower()
                    for kw in keywords:
                        if kw in content_lower and kw not in matched_keywords:
                            matched_keywords.append(kw)

            match_count = len(matched_keywords)

            if match_count == required_matches:
                print(f"📝 Resume Name: {name}")
                print(f"   ➡️ Matched keywords: {', '.join(matched_keywords)}")
                seen_names.add(name)
                found_any = True

        if not found_any:
            print(f"❌ No resumes found for exactly {required_matches} keyword(s).")

    print("\n🏁 Search complete.")

if __name__ == "__main__":
    main()
