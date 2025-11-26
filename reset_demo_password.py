#!/usr/bin/env python3
"""
Script to reset password for demo user account.
"""
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the User model from main.py
sys.path.append(os.path.dirname(__file__))
from main import User

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./korean_learning.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def reset_password(email: str, new_password: str):
    """Reset password for a user"""
    db = SessionLocal()
    try:
        # Find the user
        user = db.query(User).filter(User.email == email).first()
        if not user:
            print(f"✗ User {email} not found!")
            return None
        
        # Update password
        hashed_password = pwd_context.hash(new_password)
        user.hashed_password = hashed_password
        
        db.commit()
        db.refresh(user)
        
        print(f"✓ Successfully reset password for {email}!")
        print(f"  User ID: {user.id}")
        print(f"  New Password: {new_password}")
        print(f"  Is Premium: {user.is_premium}")
        print(f"  Is Admin: {user.is_admin}")
        
        return user
        
    except Exception as e:
        db.rollback()
        print(f"✗ Error resetting password: {str(e)}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    # Demo account credentials
    DEMO_EMAIL = "demo@example.com"
    NEW_PASSWORD = "demo123"
    
    print("Resetting password for demo account...")
    print(f"Email: {DEMO_EMAIL}")
    print(f"New Password: {NEW_PASSWORD}")
    print("-" * 50)
    
    reset_password(DEMO_EMAIL, NEW_PASSWORD)
    
    print("-" * 50)
    print("✓ Password reset complete!")
    print(f"\nYou can now login with:")
    print(f"  Email: {DEMO_EMAIL}")
    print(f"  Password: {NEW_PASSWORD}")
