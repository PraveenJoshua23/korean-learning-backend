#!/usr/bin/env python3
"""
Script to make a user an admin.
Usage: python make_admin.py <email>
"""

import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from main import User, DATABASE_URL
import os
from dotenv import load_dotenv

load_dotenv()

def make_admin(email: str):
    """Make a user an admin by their email address."""
    
    # Create database connection
    database_url = os.getenv("DATABASE_URL", "sqlite:///./korean_learning.db")
    engine = create_engine(database_url, connect_args={"check_same_thread": False} if "sqlite" in database_url else {})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    db = SessionLocal()
    
    try:
        # Find user by email
        user = db.query(User).filter(User.email == email).first()
        
        if not user:
            print(f"❌ Error: User with email '{email}' not found.")
            print("\nAvailable users:")
            all_users = db.query(User).all()
            for u in all_users:
                admin_status = "✓ ADMIN" if u.is_admin else ""
                print(f"  - {u.email} {admin_status}")
            return False
        
        # Check if already admin
        if user.is_admin:
            print(f"ℹ️  User '{email}' is already an admin.")
            return True
        
        # Make admin
        user.is_admin = True
        db.commit()
        
        print(f"✅ Success! User '{email}' is now an admin.")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        db.rollback()
        return False
    finally:
        db.close()

def list_users():
    """List all users in the database."""
    
    database_url = os.getenv("DATABASE_URL", "sqlite:///./korean_learning.db")
    engine = create_engine(database_url, connect_args={"check_same_thread": False} if "sqlite" in database_url else {})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    db = SessionLocal()
    
    try:
        users = db.query(User).all()
        
        if not users:
            print("No users found in the database.")
            return
        
        print("\n📋 All Users:")
        print("-" * 60)
        for user in users:
            admin_badge = "👑 ADMIN" if user.is_admin else ""
            premium_badge = "⭐ PREMIUM" if user.is_premium else ""
            badges = f"{admin_badge} {premium_badge}".strip()
            print(f"  {user.email}")
            if badges:
                print(f"    {badges}")
        print("-" * 60)
        
    finally:
        db.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python make_admin.py <email>          - Make a user admin")
        print("  python make_admin.py --list           - List all users")
        print("\nExample:")
        print("  python make_admin.py user@example.com")
        sys.exit(1)
    
    if sys.argv[1] == "--list":
        list_users()
    else:
        email = sys.argv[1]
        make_admin(email)
