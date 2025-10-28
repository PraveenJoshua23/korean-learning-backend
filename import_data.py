#!/usr/bin/env python3
"""
Data import script for Korean Learning Platform.
Creates sample vocabulary and grammar data, and provides CSV import functionality.
"""

import csv
import sys
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from main import Base, Vocabulary, Grammar
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./korean_learning.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_sample_vocabulary():
    """Create 80 sample Korean vocabulary items."""
    sample_vocabulary = [
        # Basic vocabulary
        ("가게", "store", "places", 1),
        ("가격", "price", "shopping", 1),
        ("가구", "furniture", "household", 2),
        ("가방", "bag", "accessories", 1),
        ("가족", "family", "relationships", 1),
        ("간식", "snack", "food", 1),
        ("감사", "gratitude", "emotions", 2),
        ("강", "river", "nature", 1),
        ("개", "dog", "animals", 1),
        ("거리", "street", "places", 1),
        
        # Intermediate vocabulary
        ("건강", "health", "health", 2),
        ("경찰", "police", "professions", 2),
        ("계절", "season", "nature", 2),
        ("고양이", "cat", "animals", 1),
        ("공원", "park", "places", 1),
        ("교육", "education", "academic", 3),
        ("교통", "transportation", "transport", 2),
        ("구름", "cloud", "weather", 2),
        ("국가", "country", "geography", 2),
        ("꽃", "flower", "nature", 1),
        
        # More vocabulary
        ("나무", "tree", "nature", 1),
        ("날씨", "weather", "weather", 1),
        ("남자", "man", "people", 1),
        ("냉장고", "refrigerator", "household", 2),
        ("녹차", "green tea", "drinks", 2),
        ("누나", "older sister", "family", 2),
        ("눈", "eye/snow", "body/weather", 1),
        ("달", "moon", "nature", 1),
        ("담배", "cigarette", "health", 2),
        ("대학교", "university", "education", 2),
        
        # Advanced vocabulary
        ("도서관", "library", "places", 2),
        ("동생", "younger sibling", "family", 2),
        ("라면", "ramen", "food", 1),
        ("레스토랑", "restaurant", "places", 1),
        ("마음", "heart/mind", "emotions", 3),
        ("만화", "comic", "entertainment", 2),
        ("머리", "head/hair", "body", 1),
        ("메시지", "message", "communication", 2),
        ("모르다", "to not know", "verbs", 2),
        ("무료", "free", "shopping", 2),
        
        # Additional vocabulary
        ("문화", "culture", "society", 3),
        ("물", "water", "drinks", 1),
        ("바다", "sea", "nature", 1),
        ("바람", "wind", "weather", 1),
        ("박물관", "museum", "places", 2),
        ("발", "foot", "body", 1),
        ("방", "room", "household", 1),
        ("법", "law", "society", 3),
        ("병원", "hospital", "places", 2),
        ("볼펜", "ballpoint pen", "stationery", 1),
        
        # More advanced vocabulary
        ("부모", "parents", "family", 2),
        ("불", "fire", "elements", 1),
        ("비", "rain", "weather", 1),
        ("사과", "apple", "food", 1),
        ("사람", "person", "people", 1),
        ("상점", "shop", "places", 1),
        ("생일", "birthday", "events", 1),
        ("선생님", "teacher", "professions", 2),
        ("세상", "world", "abstract", 3),
        ("소금", "salt", "food", 1),
        
        # Final set
        ("수업", "class", "education", 2),
        ("시간", "time", "time", 1),
        ("신문", "newspaper", "media", 2),
        ("아침", "morning", "time", 1),
        ("안경", "glasses", "accessories", 1),
        ("여자", "woman", "people", 1),
        ("영화", "movie", "entertainment", 1),
        ("오늘", "today", "time", 1),
        ("요리", "cooking", "activities", 2),
        ("우산", "umbrella", "accessories", 1),
        
        # Last 20
        ("운동", "exercise", "activities", 2),
        ("음악", "music", "entertainment", 1),
        ("의사", "doctor", "professions", 2),
        ("이름", "name", "identity", 1),
        ("자동차", "car", "transport", 1),
        ("전화", "phone", "communication", 1),
        ("집", "house", "places", 1),
        ("책", "book", "education", 1),
        ("친구", "friend", "relationships", 1),
        ("컴퓨터", "computer", "technology", 2),
        ("학교", "school", "education", 1),
        ("한국", "Korea", "geography", 1),
        ("할머니", "grandmother", "family", 2),
        ("해", "sun", "nature", 1),
        ("현금", "cash", "money", 2),
        ("호텔", "hotel", "places", 2),
        ("화장실", "bathroom", "places", 1),
        ("휴대폰", "mobile phone", "technology", 2),
        ("히터", "heater", "household", 2),
        ("힘", "strength", "abstract", 3)
    ]
    
    return sample_vocabulary

def create_sample_grammar():
    """Create 15 sample grammar points."""
    sample_grammar = [
        # Particles
        {
            "title": "은/는 (Topic Particle)",
            "category": "particles",
            "explanation": "Used to mark the topic of a sentence. 은 is used after consonants, 는 after vowels.",
            "examples": "저는 학생입니다. (I am a student.) 책은 재미있어요. (The book is interesting.)",
            "difficulty_level": 1
        },
        {
            "title": "이/가 (Subject Particle)",
            "category": "particles",
            "explanation": "Used to mark the subject of a sentence. 이 is used after consonants, 가 after vowels.",
            "examples": "고양이가 자요. (The cat sleeps.) 물이 차가워요. (The water is cold.)",
            "difficulty_level": 1
        },
        {
            "title": "을/를 (Object Particle)",
            "category": "particles",
            "explanation": "Used to mark the direct object of a sentence. 을 is used after consonants, 를 after vowels.",
            "examples": "밥을 먹어요. (I eat rice.) 음악을 들어요. (I listen to music.)",
            "difficulty_level": 1
        },
        {
            "title": "에 (Location/Time Particle)",
            "category": "particles",
            "explanation": "Used to indicate location (to/at) or specific time.",
            "examples": "학교에 가요. (I go to school.) 3시에 만나요. (Let's meet at 3 o'clock.)",
            "difficulty_level": 1
        },
        {
            "title": "에서 (Location Particle)",
            "category": "particles",
            "explanation": "Used to indicate the location where an action takes place.",
            "examples": "도서관에서 공부해요. (I study at the library.) 집에서 쉬어요. (I rest at home.)",
            "difficulty_level": 1
        },
        
        # Verb conjugations
        {
            "title": "Present Tense (-아/어요)",
            "category": "verb_conjugation",
            "explanation": "Polite present tense ending. Use -아요 after ㅏ or ㅗ vowels, -어요 after other vowels.",
            "examples": "가다 → 가요 (go), 먹다 → 먹어요 (eat), 자다 → 자요 (sleep)",
            "difficulty_level": 2
        },
        {
            "title": "Past Tense (-았/었어요)",
            "category": "verb_conjugation",
            "explanation": "Polite past tense ending. Use -았어요 after ㅏ or ㅗ vowels, -었어요 after other vowels.",
            "examples": "가다 → 갔어요 (went), 먹다 → 먹었어요 (ate), 자다 → 잤어요 (slept)",
            "difficulty_level": 2
        },
        {
            "title": "Future Tense (-ㄹ/을 거예요)",
            "category": "verb_conjugation",
            "explanation": "Used to express future actions or intentions. -ㄹ 거예요 after vowels, -을 거예요 after consonants.",
            "examples": "가다 → 갈 거예요 (will go), 먹다 → 먹을 거예요 (will eat)",
            "difficulty_level": 2
        },
        {
            "title": "Negative (-지 않다)",
            "category": "verb_conjugation",
            "explanation": "Used to make verbs negative. Attach -지 않다 to the verb stem.",
            "examples": "가다 → 가지 않아요 (don't go), 먹다 → 먹지 않아요 (don't eat)",
            "difficulty_level": 2
        },
        {
            "title": "Want to (-고 싶다)",
            "category": "verb_conjugation",
            "explanation": "Used to express desire or want. Attach -고 싶다 to the verb stem.",
            "examples": "가다 → 가고 싶어요 (want to go), 보다 → 보고 싶어요 (want to see)",
            "difficulty_level": 2
        },
        
        # Sentence structures
        {
            "title": "SOV Word Order",
            "category": "sentence_structure",
            "explanation": "Korean follows Subject-Object-Verb word order, unlike English (SVO).",
            "examples": "나는 사과를 먹어요. (I apple eat = I eat an apple.)",
            "difficulty_level": 3
        },
        {
            "title": "Questions with 까?",
            "category": "sentence_structure",
            "explanation": "Add -까요? to make polite questions from statements.",
            "examples": "가요 → 갈까요? (Shall we go?), 먹어요 → 먹을까요? (Shall we eat?)",
            "difficulty_level": 2
        },
        {
            "title": "Polite Requests (-주세요)",
            "category": "sentence_structure",
            "explanation": "Used to make polite requests. Attach -주세요 to the verb stem + 어/아.",
            "examples": "도와주세요 (Please help), 기다려주세요 (Please wait)",
            "difficulty_level": 3
        },
        {
            "title": "Connecting Sentences (-고)",
            "category": "sentence_structure",
            "explanation": "Used to connect two actions or states. Attach -고 to the verb stem.",
            "examples": "집에 가고 숙제해요. (I go home and do homework.)",
            "difficulty_level": 3
        },
        {
            "title": "Cause and Effect (-아/어서)",
            "category": "sentence_structure",
            "explanation": "Used to show cause and effect or sequence of actions.",
            "examples": "비가 와서 집에 있어요. (Because it's raining, I stay home.)",
            "difficulty_level": 3
        }
    ]
    
    return sample_grammar

def import_vocabulary_from_csv(csv_path: str, db):
    """Import vocabulary from CSV file."""
    if not Path(csv_path).exists():
        print(f"Error: CSV file '{csv_path}' does not exist.")
        return False
    
    try:
        imported_count = 0
        duplicate_count = 0
        
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Validate CSV headers
            if 'korean' not in reader.fieldnames or 'english' not in reader.fieldnames:
                print("Error: CSV must have 'korean' and 'english' columns.")
                return False
            
            for row in reader:
                korean = row['korean'].strip()
                english = row['english'].strip()
                
                if not korean or not english:
                    continue
                
                # Check if vocabulary already exists
                existing = db.query(Vocabulary).filter(Vocabulary.korean == korean).first()
                if existing:
                    duplicate_count += 1
                    continue
                
                # Determine category and difficulty (basic heuristics)
                category = row.get('category', 'imported').strip() or 'imported'
                difficulty_level = int(row.get('difficulty_level', 2)) if row.get('difficulty_level', '').isdigit() else 2
                
                vocab = Vocabulary(
                    korean=korean,
                    english=english,
                    category=category,
                    difficulty_level=difficulty_level
                )
                db.add(vocab)
                imported_count += 1
        
        db.commit()
        print(f"Successfully imported {imported_count} vocabulary items from CSV.")
        if duplicate_count > 0:
            print(f"Skipped {duplicate_count} duplicate entries.")
        
        return True
        
    except Exception as e:
        print(f"Error importing CSV: {e}")
        db.rollback()
        return False

def import_sample_data():
    """Import all sample data to the database."""
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    
    try:
        # Import vocabulary
        print("Importing sample vocabulary...")
        vocabulary_data = create_sample_vocabulary()
        
        for korean, english, category, difficulty in vocabulary_data:
            existing = db.query(Vocabulary).filter(Vocabulary.korean == korean).first()
            if not existing:
                vocab = Vocabulary(
                    korean=korean,
                    english=english,
                    category=category,
                    difficulty_level=difficulty
                )
                db.add(vocab)
        
        # Import grammar
        print("Importing sample grammar...")
        grammar_data = create_sample_grammar()
        
        for grammar_info in grammar_data:
            existing = db.query(Grammar).filter(Grammar.title == grammar_info["title"]).first()
            if not existing:
                grammar = Grammar(**grammar_info)
                db.add(grammar)
        
        db.commit()
        
        # Print statistics
        vocab_count = db.query(Vocabulary).count()
        grammar_count = db.query(Grammar).count()
        
        print(f"\nData import completed successfully!")
        print(f"Total vocabulary items: {vocab_count}")
        print(f"Total grammar points: {grammar_count}")
        
    except Exception as e:
        print(f"Error during data import: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def main():
    """Main function to handle command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Import data to Korean Learning Platform')
    parser.add_argument('--sample', action='store_true', help='Import sample vocabulary and grammar data')
    parser.add_argument('--csv', type=str, help='Import vocabulary from CSV file')
    parser.add_argument('--all', action='store_true', help='Import sample data and create initial database')
    
    args = parser.parse_args()
    
    if not any([args.sample, args.csv, args.all]):
        parser.print_help()
        return
    
    db = SessionLocal()
    
    try:
        if args.all or args.sample:
            import_sample_data()
        
        if args.csv:
            print(f"Importing vocabulary from CSV: {args.csv}")
            success = import_vocabulary_from_csv(args.csv, db)
            if not success:
                sys.exit(1)
                
    except Exception as e:
        print(f"Import failed: {e}")
        sys.exit(1)
    finally:
        db.close()

if __name__ == "__main__":
    main()