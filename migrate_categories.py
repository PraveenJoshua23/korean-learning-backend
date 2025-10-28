#!/usr/bin/env python3
"""
Script to migrate categories from vocabulary table to categories table
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from main import Base, Category, Vocabulary

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./korean_learning.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def migrate_categories():
    """Extract categories from vocabulary table and add to categories table."""
    
    # Ensure tables exist
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    
    try:
        # Get distinct categories from vocabulary table
        vocab_categories = db.query(Vocabulary.category).distinct().all()
        
        # Color mapping for categories
        colors = [
            '#3B82F6', '#10B981', '#F59E0B', '#EF4444', 
            '#8B5CF6', '#06B6D4', '#F97316', '#84CC16',
            '#EC4899', '#6B7280'
        ]
        
        added_count = 0
        skipped_count = 0
        
        for i, (category_name,) in enumerate(vocab_categories):
            if not category_name:
                continue
                
            # Check if category already exists
            existing = db.query(Category).filter(
                Category.name == category_name.title(),
                Category.type == 'vocabulary'
            ).first()
            
            if existing:
                print(f"Category '{category_name}' already exists - skipped")
                skipped_count += 1
                continue
            
            # Create new category
            category = Category(
                name=category_name.title(),
                description=f'Vocabulary related to {category_name}',
                type='vocabulary',
                color=colors[i % len(colors)],
                order=i + 1,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            db.add(category)
            added_count += 1
            print(f"Added vocabulary category: {category_name.title()}")
        
        # Add some grammar categories
        grammar_categories = [
            {
                'name': 'Particles',
                'description': 'Korean grammar particles and their usage',
                'color': '#8B5CF6'
            },
            {
                'name': 'Verb Conjugation', 
                'description': 'Korean verb endings and conjugation patterns',
                'color': '#06B6D4'
            },
            {
                'name': 'Sentence Structure',
                'description': 'Korean sentence patterns and structures', 
                'color': '#84CC16'
            }
        ]
        
        for i, gram_cat in enumerate(grammar_categories):
            # Check if category already exists
            existing = db.query(Category).filter(
                Category.name == gram_cat['name'],
                Category.type == 'grammar'
            ).first()
            
            if existing:
                print(f"Grammar category '{gram_cat['name']}' already exists - skipped")
                skipped_count += 1
                continue
            
            # Create new grammar category
            category = Category(
                name=gram_cat['name'],
                description=gram_cat['description'],
                type='grammar',
                color=gram_cat['color'],
                order=i + 1,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            db.add(category)
            added_count += 1
            print(f"Added grammar category: {gram_cat['name']}")
        
        db.commit()
        print(f"\n✅ Categories migration completed!")
        print(f"📊 Summary: {added_count} added, {skipped_count} skipped")
        
        # Show final count
        total_categories = db.query(Category).count()
        vocab_categories = db.query(Category).filter(Category.type == 'vocabulary').count()
        grammar_categories = db.query(Category).filter(Category.type == 'grammar').count()
        
        print(f"\n📈 Current categories in database:")
        print(f"   - Vocabulary: {vocab_categories}")
        print(f"   - Grammar: {grammar_categories}")
        print(f"   - Total: {total_categories}")
        
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    migrate_categories()