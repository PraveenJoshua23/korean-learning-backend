#!/usr/bin/env python3
"""
Script to create a demo user account in the Korean Learning Platform database.
"""
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the User model from main.py
sys.path.append(os.path.dirname(__file__))
from main import User, Base

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./korean_learning.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_demo_user(email: str, password: str):
    """Create a demo user account"""
    db = SessionLocal()
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == email).first()
        if existing_user:
            print(f"✓ User {email} already exists!")
            print(f"  User ID: {existing_user.id}")
            print(f"  Is Premium: {existing_user.is_premium}")
            print(f"  Is Admin: {existing_user.is_admin}")
            return existing_user
        
        # Create new user
        hashed_password = pwd_context.hash(password)
        new_user = User(
            email=email,
            hashed_password=hashed_password,
            is_premium=False,
            is_admin=False,
            current_streak=0,
            longest_streak=0,
            total_study_time=0,
            created_at=datetime.utcnow(),
            learning_level="beginner",
            daily_goal_minutes=30,
            notification_enabled=True,
            email_notifications=True,
            subscription_type="free"
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        print(f"✓ Successfully created demo user!")
        print(f"  Email: {email}")
        print(f"  Password: {password}")
        print(f"  User ID: {new_user.id}")
        print(f"  Created at: {new_user.created_at}")
        
        return new_user
        
    except Exception as e:
        db.rollback()
        print(f"✗ Error creating user: {str(e)}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    # Demo account credentials
    DEMO_EMAIL = "demo@example.com"
    DEMO_PASSWORD = "demo123"
    
    print("Creating demo user account...")
    print(f"Email: {DEMO_EMAIL}")
    print(f"Password: {DEMO_PASSWORD}")
    print("-" * 50)
    
    create_demo_user(DEMO_EMAIL, DEMO_PASSWORD)
    
    print("-" * 50)
    print("✓ Demo user setup complete!")
    print(f"\nYou can now login with:")
    print(f"  Email: {DEMO_EMAIL}")
    print(f"  Password: {DEMO_PASSWORD}")
