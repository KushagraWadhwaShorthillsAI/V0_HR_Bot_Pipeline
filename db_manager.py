import json
import uuid
from pathlib import Path
from pymongo import MongoClient
import streamlit as st  # Added for secrets access

class ResumeDBManager:
    def __init__(self):
        self.client = MongoClient(st.secrets["mongo"]["uri"])
        self.db = self.client[st.secrets["mongo"]["db_name"]]
        self.collection = self.db[st.secrets["mongo"]["collection_name"]]

    def insert_or_update_resume(self, resume: dict):
        """Upsert a resume based on name and email.
        
        If a resume with the same name and email exists, it will be replaced.
        Otherwise, a new resume will be inserted.
        """
        # Create a query to find existing resume by name and email
        # It's ok for values to be null, so we need to handle that case
        query = {}
        
        # Only add fields to query if they exist and are not empty
        if resume.get("name") and resume.get("email"):
            query = {"name": resume.get("name"), "email": resume.get("email")}
        elif resume.get("name"):
            query = {"name": resume.get("name")}
        elif resume.get("email"):
            query = {"email": resume.get("email")}
        
        # If we have a valid query, try to upsert
        if query:                
            # Perform the upsert operation
            result = self.collection.replace_one(query, resume, upsert=True)
            
            if result.matched_count:
                print(f"✅ Updated existing resume for {resume.get('name', 'Unknown')} ({resume.get('email', 'No email')}) with ID: {resume.get('_id')}")
                return resume.get("_id")
            else:
                print(f"✅ Inserted new resume for {resume.get('name', 'Unknown')} ({resume.get('email', 'No email')}) with ID: {resume.get('_id')}")
                return resume.get("_id")
        else:
            # If we don't have a valid query, just insert with a new ID
            if "_id" not in resume:
                resume["_id"] = str(uuid.uuid4())
            result = self.collection.insert_one(resume)
            print(f"✅ Inserted document with new ID: {result.inserted_id}")
            return result.inserted_id
        
    def bulk_insert(self, folder_path: str):
        """Upsert all JSON files in a folder using insert_or_update_resume logic."""
        folder = Path(folder_path)
        files = list(folder.glob("*.json"))
        print(f"📂 Found {len(files)} resumes to insert or update.\n")

        inserted, failed = 0, 0

        for file in files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                # Use the same logic as the single resume insertion
                self.insert_or_update_resume(doc)
                inserted += 1
            except Exception as e:
                print(f"❌ Failed to upsert {file.name}: {e}")
                failed += 1

        print(f"\n📊 Summary: Total = {len(files)}, Upserted = {inserted}, Failed = {failed}")

    def find(self, query: dict):
        """Find resumes matching a query."""
        print(f"🔍 Finding resumes matching: {query}")
        results = list(self.collection.find(query))
        print(f"🔎 Found {len(results)} resumes.\n")
        for res in results:
            print(f"- {res.get('name')} | {res.get('email')} | ID: {res.get('_id')}")
        return results

    def update_resume(self, update_data: dict):
        """Update a resume by _id."""
        _id = update_data.pop("_id", None)
        if not _id:
            print("❌ Update failed: '_id' field is required.")
            return None
        result = self.collection.update_one({"_id": _id}, {"$set": update_data})
        if result.modified_count:
            print(f"✅ Updated resume with ID {_id}")
        else:
            print(f"⚠️ No resume found or no change for ID {_id}")
        return result

    def delete_resume(self, delete_data: dict):
        """Delete a resume by _id."""
        _id = delete_data.get("_id")
        if not _id:
            print("❌ Delete failed: '_id' field is required.")
            return None
        
        # Ensure we're using just the ID string, not an object
        result = self.collection.delete_one({"_id": _id})
        if result.deleted_count:
            print(f"🗑️ Deleted resume with ID {_id}")
        else:
            print(f"⚠️ No resume found with ID {_id}")
        return result
        
    def delete_all_resumes(self):
        """Delete all resumes in the collection."""
        result = self.collection.delete_many({})
        print(f"🗑️ Deleted {result.deleted_count} resumes.")
        return result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Path to single resume JSON file")
    parser.add_argument("--folder", help="Path to folder containing multiple JSON files")
    parser.add_argument("--find", help="Find query in JSON format")
    parser.add_argument("--update", help="JSON string with _id and fields to update")
    parser.add_argument("--delete", help="JSON string with _id of resume to delete")
    parser.add_argument("--delete-all", action="store_true", help="Delete all resumes in the collection")

    args = parser.parse_args()
    db = ResumeDBManager()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            doc = json.load(f)
            db.insert_or_update_resume(doc)

    elif args.folder:
        db.bulk_insert(args.folder)

    elif args.find:
        try:
            query = json.loads(args.find)
            db.find(query)
        except Exception as e:
            print(f"❌ Invalid JSON for --find: {e}")

    elif args.update:
        try:
            update_data = json.loads(args.update)
            db.update_resume(update_data)
        except Exception as e:
            print(f"❌ Invalid JSON for --update: {e}")

    elif args.delete:
        try:
            delete_data = json.loads(args.delete)
            db.delete_resume(delete_data)
        except Exception as e:
            print(f"❌ Invalid JSON for --delete: {e}")
    elif args.delete_all:
        db.delete_all_resumes()

    else:
        print("⚠️ Please provide one of --file, --folder, --find, --update, or --delete.")