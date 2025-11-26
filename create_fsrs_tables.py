#!/usr/bin/env python3
"""
FSRS Database Schema Migration
Creates tables for Free Spaced Repetition Scheduler algorithm
Supports both vocabulary and grammar content types
"""

import sqlite3
from datetime import datetime
import os

def create_fsrs_tables():
    # Get database path
    db_path = os.path.join(os.path.dirname(__file__), 'korean_learning.db')
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # FSRS Cards table - stores spaced repetition data for each user-content pair
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fsrs_cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            content_type VARCHAR(20) NOT NULL CHECK (content_type IN ('vocabulary', 'grammar')),
            content_id INTEGER NOT NULL,
            
            -- FSRS Core Parameters
            stability REAL NOT NULL DEFAULT 1.0,          -- Memory strength in days
            difficulty REAL NOT NULL DEFAULT 5.0,         -- Intrinsic difficulty (0-10)
            retrievability REAL NOT NULL DEFAULT 0.9,     -- Current recall probability (0-1)
            
            -- Card State
            state VARCHAR(20) NOT NULL DEFAULT 'new' CHECK (state IN ('new', 'learning', 'review', 'relearning')),
            due_date DATETIME NOT NULL,
            last_review DATETIME NULL,
            
            -- Scheduling Info
            interval_days INTEGER NOT NULL DEFAULT 1,     -- Current interval in days
            review_count INTEGER NOT NULL DEFAULT 0,      -- Total number of reviews
            lapse_count INTEGER NOT NULL DEFAULT 0,       -- Number of times forgotten
            
            -- Metadata
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            
            -- Constraints
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
            UNIQUE (user_id, content_type, content_id)
        )
    ''')
    
    # FSRS Review History table - tracks all review sessions for analytics
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fsrs_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            content_type VARCHAR(20) NOT NULL CHECK (content_type IN ('vocabulary', 'grammar')),
            content_id INTEGER NOT NULL,
            card_id INTEGER NOT NULL,
            
            -- Review Data
            review_date DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            grade INTEGER NOT NULL CHECK (grade IN (1, 2, 3, 4)), -- Again, Hard, Good, Easy
            response_time_ms INTEGER NULL,                         -- Time taken to respond
            
            -- FSRS State Before Review
            previous_state VARCHAR(20) NOT NULL,
            previous_stability REAL NOT NULL,
            previous_difficulty REAL NOT NULL,
            previous_interval INTEGER NOT NULL,
            
            -- FSRS State After Review
            new_state VARCHAR(20) NOT NULL,
            new_stability REAL NOT NULL,
            new_difficulty REAL NOT NULL,
            new_interval INTEGER NOT NULL,
            new_due_date DATETIME NOT NULL,
            
            -- Constraints
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
            FOREIGN KEY (card_id) REFERENCES fsrs_cards (id) ON DELETE CASCADE
        )
    ''')
    
    # Study Sessions table - tracks study session metadata
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fsrs_study_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            content_type VARCHAR(20) NOT NULL CHECK (content_type IN ('vocabulary', 'grammar', 'mixed')),
            
            -- Session Info
            session_type VARCHAR(20) NOT NULL DEFAULT 'review' CHECK (session_type IN ('new', 'review', 'mixed', 'cram')),
            started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            ended_at DATETIME NULL,
            duration_seconds INTEGER NULL,
            
            -- Session Stats
            cards_reviewed INTEGER NOT NULL DEFAULT 0,
            cards_learned INTEGER NOT NULL DEFAULT 0,       -- New cards that graduated
            cards_relearned INTEGER NOT NULL DEFAULT 0,     -- Cards that came out of relearning
            average_grade REAL NULL,                        -- Average grade for the session
            
            -- Performance Metrics
            total_response_time_ms INTEGER NULL,
            average_response_time_ms INTEGER NULL,
            
            -- Constraints
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    # User FSRS Preferences table - stores user-specific algorithm settings
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fsrs_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL UNIQUE,
            
            -- Daily Limits
            new_cards_per_day INTEGER NOT NULL DEFAULT 10,
            max_reviews_per_day INTEGER NOT NULL DEFAULT 50,
            
            -- Algorithm Parameters (can be customized per user)
            initial_stability REAL NOT NULL DEFAULT 2.0,
            learning_steps TEXT NOT NULL DEFAULT '1,10',    -- Minutes for learning steps
            graduating_interval INTEGER NOT NULL DEFAULT 1, -- Days
            easy_interval INTEGER NOT NULL DEFAULT 4,       -- Days
            
            -- Study Preferences
            preferred_study_time TIME NULL,
            reminder_enabled BOOLEAN NOT NULL DEFAULT TRUE,
            auto_advance BOOLEAN NOT NULL DEFAULT FALSE,
            
            -- Algorithm Weights (for personalization)
            w_stability_growth REAL NOT NULL DEFAULT 1.0,
            w_difficulty_decay REAL NOT NULL DEFAULT 1.0,
            
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    # Create indexes for performance
    indexes = [
        'CREATE INDEX IF NOT EXISTS idx_fsrs_cards_user_content ON fsrs_cards (user_id, content_type, content_id)',
        'CREATE INDEX IF NOT EXISTS idx_fsrs_cards_due_date ON fsrs_cards (due_date, user_id)',
        'CREATE INDEX IF NOT EXISTS idx_fsrs_cards_state ON fsrs_cards (state, user_id)',
        'CREATE INDEX IF NOT EXISTS idx_fsrs_reviews_user_date ON fsrs_reviews (user_id, review_date)',
        'CREATE INDEX IF NOT EXISTS idx_fsrs_reviews_card ON fsrs_reviews (card_id)',
        'CREATE INDEX IF NOT EXISTS idx_fsrs_sessions_user ON fsrs_study_sessions (user_id, started_at)',
    ]
    
    for index in indexes:
        cursor.execute(index)
    
    # Create triggers to update timestamps
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS update_fsrs_cards_timestamp 
        AFTER UPDATE ON fsrs_cards 
        BEGIN
            UPDATE fsrs_cards SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END
    ''')
    
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS update_fsrs_preferences_timestamp 
        AFTER UPDATE ON fsrs_preferences 
        BEGIN
            UPDATE fsrs_preferences SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END
    ''')
    
    conn.commit()
    conn.close()
    
    print("✅ FSRS database tables created successfully!")
    print("📋 Created tables:")
    print("  - fsrs_cards: Spaced repetition card data")
    print("  - fsrs_reviews: Review history for analytics")
    print("  - fsrs_study_sessions: Study session tracking")
    print("  - fsrs_preferences: User algorithm preferences")

if __name__ == "__main__":
    create_fsrs_tables()