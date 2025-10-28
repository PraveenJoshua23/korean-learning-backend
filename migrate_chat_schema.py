#!/usr/bin/env python3
"""
Migration script to move from old chat schema to new optimized schema.

This script will:
1. Create new tables (conversations, messages)
2. Migrate data from chat_conversations to the new schema
3. Preserve all existing data
4. Provide rollback functionality
"""

import os
import sys
import json
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

# Add the parent directory to the path so we can import from main.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./korean_learning.db")

# Create engine and session
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_backup():
    """Create a backup of the existing chat tables"""
    print("Creating backup of existing chat tables...")
    
    with engine.connect() as conn:
        # Backup chat_conversations table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chat_conversations_backup AS 
            SELECT * FROM chat_conversations
        """))
        
        # Backup chat_messages table if it exists
        try:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS chat_messages_backup AS 
                SELECT * FROM chat_messages
            """))
        except:
            print("chat_messages table doesn't exist or is empty, skipping backup")
        
        conn.commit()
    
    print("✅ Backup completed")

def create_new_tables():
    """Create the new optimized chat tables"""
    print("Creating new optimized chat tables...")
    
    with engine.connect() as conn:
        # Create conversations table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                title VARCHAR(255),
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                message_count INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """))
        
        # Create messages table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                conversation_id INTEGER NOT NULL,
                role VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                content_length INTEGER,
                token_count INTEGER,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """))
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at)",
            "CREATE INDEX IF NOT EXISTS idx_conversations_is_active ON conversations(is_active)",
            "CREATE INDEX IF NOT EXISTS idx_conversations_user_active_updated ON conversations(user_id, is_active, updated_at)",
            "CREATE INDEX IF NOT EXISTS idx_conversations_user_created ON conversations(user_id, created_at)",
            
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role)",
            "CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation_created ON messages(conversation_id, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation_role ON messages(conversation_id, role)",
        ]
        
        for index_sql in indexes:
            conn.execute(text(index_sql))
        
        conn.commit()
    
    print("✅ New tables and indexes created")

def migrate_data():
    """Migrate data from old schema to new schema"""
    print("Migrating data from old schema to new schema...")
    
    db = SessionLocal()
    try:
        # Get all chat conversations from the old table
        result = db.execute(text("SELECT * FROM chat_conversations ORDER BY created_at"))
        old_conversations = result.fetchall()
        
        if not old_conversations:
            print("No data to migrate")
            return
        
        print(f"Found {len(old_conversations)} conversations to migrate")
        
        # Group conversations by user to create conversation sessions
        user_conversations = {}
        for conv in old_conversations:
            user_id = conv.user_id
            if user_id not in user_conversations:
                user_conversations[user_id] = []
            user_conversations[user_id].append(conv)
        
        total_migrated = 0
        
        for user_id, conversations in user_conversations.items():
            print(f"Migrating {len(conversations)} conversations for user {user_id}")
            
            # For each user, create conversation sessions
            # Group by date to create logical conversation groups
            current_conversation_id = None
            current_date = None
            message_count = 0
            
            for conv in conversations:
                conv_date = conv.created_at.date() if hasattr(conv.created_at, 'date') else conv.created_at[:10]
                
                # Create a new conversation if:
                # 1. This is the first conversation
                # 2. The date changed (new day = new conversation)
                # 3. More than 4 hours passed (logical conversation break)
                if (current_conversation_id is None or 
                    current_date != conv_date or 
                    message_count >= 20):  # Limit messages per conversation
                    
                    # Create new conversation
                    title = f"Chat Session {conv_date}" if conv_date != current_date else f"Chat Session {conv_date} (cont.)"
                    
                    conversation_result = db.execute(text("""
                        INSERT INTO conversations (user_id, title, created_at, updated_at, is_active, message_count)
                        VALUES (:user_id, :title, :created_at, :updated_at, 1, 0)
                    """), {
                        'user_id': user_id,
                        'title': title,
                        'created_at': conv.created_at,
                        'updated_at': conv.created_at
                    })
                    
                    current_conversation_id = conversation_result.lastrowid
                    current_date = conv_date
                    message_count = 0
                    print(f"  Created conversation {current_conversation_id}: {title}")
                
                # Insert user message
                db.execute(text("""
                    INSERT INTO messages (conversation_id, role, content, created_at, content_length)
                    VALUES (:conversation_id, 'user', :content, :created_at, :content_length)
                """), {
                    'conversation_id': current_conversation_id,
                    'content': conv.message,
                    'created_at': conv.created_at,
                    'content_length': len(conv.message)
                })
                
                # Insert assistant response
                db.execute(text("""
                    INSERT INTO messages (conversation_id, role, content, created_at, content_length)
                    VALUES (:conversation_id, 'assistant', :content, :created_at, :content_length)
                """), {
                    'conversation_id': current_conversation_id,
                    'content': conv.response,
                    'created_at': conv.created_at,
                    'content_length': len(conv.response)
                })
                
                message_count += 2  # user + assistant message
                total_migrated += 1
                
                # Update conversation's updated_at and message_count
                db.execute(text("""
                    UPDATE conversations 
                    SET updated_at = :updated_at, message_count = :message_count
                    WHERE id = :conversation_id
                """), {
                    'conversation_id': current_conversation_id,
                    'updated_at': conv.created_at,
                    'message_count': message_count
                })
        
        db.commit()
        print(f"✅ Successfully migrated {total_migrated} conversation pairs")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Migration failed: {str(e)}")
        raise
    finally:
        db.close()

def verify_migration():
    """Verify that the migration was successful"""
    print("Verifying migration...")
    
    db = SessionLocal()
    try:
        # Count records in old and new tables
        old_count = db.execute(text("SELECT COUNT(*) FROM chat_conversations")).scalar()
        new_conversations = db.execute(text("SELECT COUNT(*) FROM conversations")).scalar()
        new_messages = db.execute(text("SELECT COUNT(*) FROM messages")).scalar()
        
        print(f"Original conversations: {old_count}")
        print(f"New conversations: {new_conversations}")
        print(f"New messages: {new_messages}")
        print(f"Expected messages: {old_count * 2}")  # Each old conversation = 2 messages
        
        if new_messages == old_count * 2:
            print("✅ Migration verification successful!")
            return True
        else:
            print("❌ Migration verification failed - message count mismatch")
            return False
            
    finally:
        db.close()

def rename_old_tables():
    """Rename old tables to mark them as deprecated"""
    print("Renaming old tables...")
    
    with engine.connect() as conn:
        # Rename tables to indicate they're deprecated
        conn.execute(text("ALTER TABLE chat_conversations RENAME TO chat_conversations_deprecated"))
        try:
            conn.execute(text("ALTER TABLE chat_messages RENAME TO chat_messages_deprecated"))
        except:
            print("chat_messages table doesn't exist, skipping rename")
        
        conn.commit()
    
    print("✅ Old tables renamed with '_deprecated' suffix")

def rollback_migration():
    """Rollback the migration if something goes wrong"""
    print("Rolling back migration...")
    
    with engine.connect() as conn:
        # Drop new tables
        conn.execute(text("DROP TABLE IF EXISTS messages"))
        conn.execute(text("DROP TABLE IF EXISTS conversations"))
        
        # Restore original tables from backup
        conn.execute(text("ALTER TABLE chat_conversations_backup RENAME TO chat_conversations"))
        try:
            conn.execute(text("ALTER TABLE chat_messages_backup RENAME TO chat_messages"))
        except:
            pass
        
        conn.commit()
    
    print("✅ Migration rolled back successfully")

def main():
    """Main migration function"""
    print("🚀 Starting chat schema migration...")
    print("=" * 50)
    
    try:
        # Step 1: Create backup
        create_backup()
        
        # Step 2: Create new tables
        create_new_tables()
        
        # Step 3: Migrate data
        migrate_data()
        
        # Step 4: Verify migration
        if verify_migration():
            # Step 5: Rename old tables
            rename_old_tables()
            print("\n🎉 Migration completed successfully!")
            print("Old tables have been renamed with '_deprecated' suffix")
            print("You can safely delete them after testing the new system")
        else:
            print("\n❌ Migration verification failed")
            response = input("Do you want to rollback? (y/N): ")
            if response.lower() == 'y':
                rollback_migration()
            
    except Exception as e:
        print(f"\n❌ Migration failed with error: {str(e)}")
        response = input("Do you want to rollback? (y/N): ")
        if response.lower() == 'y':
            rollback_migration()
        else:
            print("Migration left in partial state - manual cleanup may be required")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate chat schema to optimized version')
    parser.add_argument('--rollback', action='store_true', help='Rollback the migration')
    parser.add_argument('--verify', action='store_true', help='Only verify the migration')
    
    args = parser.parse_args()
    
    if args.rollback:
        rollback_migration()
    elif args.verify:
        verify_migration()
    else:
        main()