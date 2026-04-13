import json
import os
from datetime import datetime
from typing import List, Optional, Union

from langchain_core.tools import tool

DATA_DIR = "data"
DB_FILE = os.path.join(DATA_DIR, "knowledge_base.json")


@tool
def save_to_knowledge_base(title: str, url: str, summary: str, tags: Optional[Union[str, List[str]]] = None) -> str:
    """Save a technical article and its summary to the local knowledge base."""
    # Ensure directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Process tags
    processed_tags = []
    if tags:
        if isinstance(tags, str):
            processed_tags = [tag.strip() for tag in tags.split(",")]
        else:
            processed_tags = [str(tag).strip() for tag in tags]

    # Prepare the entry
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "title": title,
        "url": url,
        "summary": summary,
        "tags": processed_tags
    }

    # Load existing data
    data = []
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []

    # Append new entry
    data.append(entry)

    # Save back to file
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    return f"Successfully saved '{title}' to the knowledge base."
