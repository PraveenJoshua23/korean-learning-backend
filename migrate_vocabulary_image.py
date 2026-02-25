#!/usr/bin/env python3
"""
Script to add image_url column to the vocabulary table.
Run this once to migrate the existing database.
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./korean_learning.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})

def migrate():
    """Add image_url column to vocabulary table if it doesn't exist."""
    with engine.connect() as conn:
        # Check if column already exists
        result = conn.execute(text("PRAGMA table_info(vocabulary)"))
        columns = [row[1] for row in result.fetchall()]
        
        if "image_url" in columns:
            print("✅ Column 'image_url' already exists in vocabulary table. No migration needed.")
            return
        
        # Add the column
        conn.execute(text("ALTER TABLE vocabulary ADD COLUMN image_url VARCHAR"))
        conn.commit()
        print("✅ Successfully added 'image_url' column to vocabulary table.")

if __name__ == "__main__":
    migrate()
