"""
FSRS API Endpoints
Provides RESTful API for Free Spaced Repetition Scheduler functionality
"""

from fastapi import HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from fsrs_algorithm import FSRSAlgorithm, FSRSCard as AlgoCard, ReviewResult as AlgoReview, Grade, CardState

logger = logging.getLogger(__name__)

# Initialize FSRS Algorithm
fsrs_algo = FSRSAlgorithm()

def setup_fsrs_endpoints(app, Base, get_db, get_current_user, User, Vocabulary, Grammar):
    """Setup FSRS endpoints with dependencies injected"""
    
    # =============================================
    # FSRS Database Models
    # =============================================
    
    class FSRSCard(Base):
        __tablename__ = "fsrs_cards"
        
        id = Column(Integer, primary_key=True, index=True)
        user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
        content_type = Column(String, nullable=False)  # 'vocabulary' or 'grammar'
        content_id = Column(Integer, nullable=False)
        
        # FSRS Core Parameters
        stability = Column(Float, nullable=False, default=2.0)
        difficulty = Column(Float, nullable=False, default=5.0)
        retrievability = Column(Float, nullable=False, default=0.9)
        
        # Card State
        state = Column(String, nullable=False, default="new")
        due_date = Column(DateTime, nullable=False)
        last_review = Column(DateTime, nullable=True)
        
        # Scheduling Info
        interval_days = Column(Integer, nullable=False, default=1)
        review_count = Column(Integer, nullable=False, default=0)
        lapse_count = Column(Integer, nullable=False, default=0)
        
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow)

    class FSRSReview(Base):
        __tablename__ = "fsrs_reviews"
        
        id = Column(Integer, primary_key=True, index=True)
        user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
        content_type = Column(String, nullable=False)
        content_id = Column(Integer, nullable=False)
        card_id = Column(Integer, ForeignKey("fsrs_cards.id"), nullable=False)
        
        review_date = Column(DateTime, nullable=False, default=datetime.utcnow)
        grade = Column(Integer, nullable=False)
        response_time_ms = Column(Integer, nullable=True)
        
        # FSRS State Before Review
        previous_state = Column(String, nullable=False)
        previous_stability = Column(Float, nullable=False)
        previous_difficulty = Column(Float, nullable=False)
        previous_interval = Column(Integer, nullable=False)
        
        # FSRS State After Review
        new_state = Column(String, nullable=False)
        new_stability = Column(Float, nullable=False)
        new_difficulty = Column(Float, nullable=False)
        new_interval = Column(Integer, nullable=False)
        new_due_date = Column(DateTime, nullable=False)

    class FSRSStudySession(Base):
        __tablename__ = "fsrs_study_sessions"
        
        id = Column(Integer, primary_key=True, index=True)
        user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
        content_type = Column(String, nullable=False)
        session_type = Column(String, nullable=False, default="review")
        
        started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
        ended_at = Column(DateTime, nullable=True)
        duration_seconds = Column(Integer, nullable=True)
        
        cards_reviewed = Column(Integer, nullable=False, default=0)
        cards_learned = Column(Integer, nullable=False, default=0)
        cards_relearned = Column(Integer, nullable=False, default=0)
        average_grade = Column(Float, nullable=True)
        
        total_response_time_ms = Column(Integer, nullable=True)
        average_response_time_ms = Column(Integer, nullable=True)

    class FSRSPreferences(Base):
        __tablename__ = "fsrs_preferences"
        
        id = Column(Integer, primary_key=True, index=True)
        user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
        
        new_cards_per_day = Column(Integer, nullable=False, default=10)
        max_reviews_per_day = Column(Integer, nullable=False, default=50)
        
        initial_stability = Column(Float, nullable=False, default=2.0)
        learning_steps = Column(String, nullable=False, default="1,10")
        graduating_interval = Column(Integer, nullable=False, default=1)
        easy_interval = Column(Integer, nullable=False, default=4)
        
        preferred_study_time = Column(String, nullable=True)
        reminder_enabled = Column(Boolean, nullable=False, default=True)
        auto_advance = Column(Boolean, nullable=False, default=False)
        
        w_stability_growth = Column(Float, nullable=False, default=1.0)
        w_difficulty_decay = Column(Float, nullable=False, default=1.0)
        
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow)

    # =============================================
    # Pydantic Models
    # =============================================
    
    class FSRSCardResponse(BaseModel):
        id: int
        user_id: int
        content_type: str
        content_id: int
        stability: float
        difficulty: float
        retrievability: float
        state: str
        due_date: datetime
        last_review: Optional[datetime]
        interval_days: int
        review_count: int
        lapse_count: int
        created_at: datetime
        updated_at: datetime
        
        class Config:
            from_attributes = True

    class StudyCardResponse(BaseModel):
        fsrs_card: FSRSCardResponse
        content: dict

    class StartStudySessionRequest(BaseModel):
        content_type: str
        session_type: str = "review"
        max_new_cards: Optional[int] = 10
        max_review_cards: Optional[int] = 50
        categories: Optional[List[str]] = None

    class ReviewSubmissionRequest(BaseModel):
        session_id: int
        card_id: int
        grade: int
        response_time_ms: Optional[int] = None

    # =============================================
    # Helper Functions
    # =============================================
    
    def get_or_create_fsrs_card(db: Session, user_id: int, content_type: str, content_id: int) -> FSRSCard:
        """Get existing FSRS card or create a new one"""
        card = db.query(FSRSCard).filter_by(
            user_id=user_id,
            content_type=content_type,
            content_id=content_id
        ).first()
        
        if not card:
            card = FSRSCard(
                user_id=user_id,
                content_type=content_type,
                content_id=content_id,
                due_date=datetime.utcnow()
            )
            db.add(card)
            db.commit()
            db.refresh(card)
        
        return card

    def convert_to_algo_card(db_card: FSRSCard) -> AlgoCard:
        """Convert database card to algorithm card"""
        return AlgoCard(
            id=db_card.id,
            user_id=db_card.user_id,
            content_type=db_card.content_type,
            content_id=db_card.content_id,
            stability=db_card.stability,
            difficulty=db_card.difficulty,
            retrievability=db_card.retrievability,
            state=CardState(db_card.state),
            due_date=db_card.due_date,
            last_review=db_card.last_review,
            interval_days=db_card.interval_days,
            review_count=db_card.review_count,
            lapse_count=db_card.lapse_count
        )

    def update_db_card_from_algo(db_card: FSRSCard, algo_card: AlgoCard) -> None:
        """Update database card with algorithm results"""
        db_card.stability = algo_card.stability
        db_card.difficulty = algo_card.difficulty
        db_card.retrievability = algo_card.retrievability
        db_card.state = algo_card.state.value
        db_card.due_date = algo_card.due_date
        db_card.last_review = algo_card.last_review
        db_card.interval_days = algo_card.interval_days
        db_card.review_count = algo_card.review_count
        db_card.lapse_count = algo_card.lapse_count
        db_card.updated_at = datetime.utcnow()

    # =============================================
    # API Endpoints
    # =============================================

    @app.post("/api/fsrs/session/start")
    async def start_study_session(
        request: StartStudySessionRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Start a new study session"""
        try:
            # Create study session
            session = FSRSStudySession(
                user_id=current_user.id,
                content_type=request.content_type,
                session_type=request.session_type
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            
            # Get due cards
            now = datetime.utcnow()
            
            # Query for due review cards
            due_cards_query = db.query(FSRSCard).filter(
                FSRSCard.user_id == current_user.id,
                FSRSCard.content_type == request.content_type,
                FSRSCard.due_date <= now,
                FSRSCard.state.in_(["learning", "review", "relearning"])
            ).order_by(FSRSCard.due_date)
            
            if request.max_review_cards:
                due_cards_query = due_cards_query.limit(request.max_review_cards)
            
            due_cards = due_cards_query.all()
            
            # Get new cards if needed
            new_cards = []
            if request.session_type in ["new", "mixed"] and request.max_new_cards and request.max_new_cards > 0:
                # Find content that doesn't have FSRS cards yet
                if request.content_type == "vocabulary":
                    existing_content_ids = db.query(FSRSCard.content_id).filter(
                        FSRSCard.user_id == current_user.id,
                        FSRSCard.content_type == "vocabulary"
                    ).subquery()
                    
                    new_vocab = db.query(Vocabulary).filter(
                        ~Vocabulary.id.in_(existing_content_ids)
                    ).limit(request.max_new_cards).all()
                    
                    for vocab in new_vocab:
                        card = get_or_create_fsrs_card(db, current_user.id, "vocabulary", vocab.id)
                        new_cards.append(card)
                
                elif request.content_type == "grammar":
                    existing_content_ids = db.query(FSRSCard.content_id).filter(
                        FSRSCard.user_id == current_user.id,
                        FSRSCard.content_type == "grammar"
                    ).subquery()
                    
                    new_grammar = db.query(Grammar).filter(
                        ~Grammar.id.in_(existing_content_ids)
                    ).limit(request.max_new_cards).all()
                    
                    for grammar in new_grammar:
                        card = get_or_create_fsrs_card(db, current_user.id, "grammar", grammar.id)
                        new_cards.append(card)
            
            # Combine and prepare study cards
            all_cards = due_cards + new_cards
            study_cards = []
            
            for card in all_cards:
                if request.content_type == "vocabulary":
                    content = db.query(Vocabulary).filter_by(id=card.content_id).first()
                    if content:
                        content_dict = {
                            "id": content.id,
                            "korean": content.korean,
                            "english": content.english,
                            "pronunciation": content.pronunciation,
                            "examples": content.examples.split("|||") if content.examples else [],
                            "category": content.category,
                            "difficulty_level": content.difficulty_level
                        }
                        
                        # Convert card to dict for proper JSON serialization
                        card_dict = {
                            "id": card.id,
                            "user_id": card.user_id,
                            "content_type": card.content_type,
                            "content_id": card.content_id,
                            "stability": card.stability,
                            "difficulty": card.difficulty,
                            "retrievability": card.retrievability,
                            "state": card.state,
                            "due_date": card.due_date.isoformat() if card.due_date else None,
                            "last_review": card.last_review.isoformat() if card.last_review else None,
                            "interval_days": card.interval_days,
                            "review_count": card.review_count,
                            "lapse_count": card.lapse_count,
                            "created_at": card.created_at.isoformat() if hasattr(card, 'created_at') and card.created_at else None,
                            "updated_at": card.updated_at.isoformat() if hasattr(card, 'updated_at') and card.updated_at else None
                        }
                        
                        study_cards.append({
                            "fsrs_card": card_dict,
                            "content": content_dict
                        })
                
                elif request.content_type == "grammar":
                    content = db.query(Grammar).filter_by(id=card.content_id).first()
                    if content:
                        content_dict = {
                            "id": content.id,
                            "title": content.title,
                            "explanation": content.explanation,
                            "examples": content.examples.split("|||") if content.examples else [],
                            "category": content.category,
                            "difficulty_level": content.difficulty_level
                        }
                        
                        # Convert card to dict for proper JSON serialization
                        card_dict = {
                            "id": card.id,
                            "user_id": card.user_id,
                            "content_type": card.content_type,
                            "content_id": card.content_id,
                            "stability": card.stability,
                            "difficulty": card.difficulty,
                            "retrievability": card.retrievability,
                            "state": card.state,
                            "due_date": card.due_date.isoformat() if card.due_date else None,
                            "last_review": card.last_review.isoformat() if card.last_review else None,
                            "interval_days": card.interval_days,
                            "review_count": card.review_count,
                            "lapse_count": card.lapse_count,
                            "created_at": card.created_at.isoformat() if hasattr(card, 'created_at') and card.created_at else None,
                            "updated_at": card.updated_at.isoformat() if hasattr(card, 'updated_at') and card.updated_at else None
                        }
                        
                        study_cards.append({
                            "fsrs_card": card_dict,
                            "content": content_dict
                        })
            
            # Calculate stats
            total_due = db.query(FSRSCard).filter(
                FSRSCard.user_id == current_user.id,
                FSRSCard.content_type == request.content_type,
                FSRSCard.due_date <= now
            ).count()
            
            stats = {
                "total_cards": len(study_cards),
                "due_cards": len(due_cards),
                "new_cards": len(new_cards),
                "retention_rate": 0.9,  # TODO: Calculate from review history
                "study_streak": 0  # TODO: Calculate from session history
            }
            
            # Convert session to dict for proper JSON serialization
            session_dict = {
                "id": session.id,
                "user_id": session.user_id,
                "content_type": session.content_type,
                "session_type": session.session_type,
                "started_at": session.started_at.isoformat() if session.started_at else None,
                "ended_at": session.ended_at.isoformat() if session.ended_at else None,
                "duration_seconds": session.duration_seconds,
                "cards_reviewed": session.cards_reviewed,
                "cards_learned": session.cards_learned,
                "cards_relearned": session.cards_relearned
            }
            
            return {
                "session": session_dict,
                "cards": study_cards,
                "stats": stats
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Start study session error: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to start study session")

    @app.post("/api/fsrs/session/review")
    async def submit_review(
        request: ReviewSubmissionRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Submit a review for a card"""
        try:
            # Get the card and session
            card = db.query(FSRSCard).filter_by(
                id=request.card_id,
                user_id=current_user.id
            ).first()
            
            if not card:
                raise HTTPException(status_code=404, detail="Card not found")
            
            session = db.query(FSRSStudySession).filter_by(
                id=request.session_id,
                user_id=current_user.id
            ).first()
            
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            # Store previous state for review history
            previous_state = card.state
            previous_stability = card.stability
            previous_difficulty = card.difficulty
            previous_interval = card.interval_days
            
            # Convert to algorithm format and schedule
            algo_card = convert_to_algo_card(card)
            review_result = AlgoReview(grade=Grade(request.grade))
            updated_algo_card = fsrs_algo.schedule_card(algo_card, review_result)
            
            # Update database card
            update_db_card_from_algo(card, updated_algo_card)
            
            # Create review history record
            review_record = FSRSReview(
                user_id=current_user.id,
                content_type=card.content_type,
                content_id=card.content_id,
                card_id=card.id,
                grade=request.grade,
                response_time_ms=request.response_time_ms,
                previous_state=previous_state,
                previous_stability=previous_stability,
                previous_difficulty=previous_difficulty,
                previous_interval=previous_interval,
                new_state=card.state,
                new_stability=card.stability,
                new_difficulty=card.difficulty,
                new_interval=card.interval_days,
                new_due_date=card.due_date
            )
            db.add(review_record)
            
            # Update session stats
            session.cards_reviewed += 1
            if card.state in ["review", "learning"] and previous_state == "new":
                session.cards_learned += 1
            if card.state == "review" and previous_state == "relearning":
                session.cards_relearned += 1
            
            db.commit()
            
            # Convert updated card to dict for proper JSON serialization
            updated_card_dict = {
                "id": card.id,
                "user_id": card.user_id,
                "content_type": card.content_type,
                "content_id": card.content_id,
                "stability": card.stability,
                "difficulty": card.difficulty,
                "retrievability": card.retrievability,
                "state": card.state,
                "due_date": card.due_date.isoformat() if card.due_date else None,
                "last_review": card.last_review.isoformat() if card.last_review else None,
                "interval_days": card.interval_days,
                "review_count": card.review_count,
                "lapse_count": card.lapse_count,
                "created_at": card.created_at.isoformat() if hasattr(card, 'created_at') and card.created_at else None,
                "updated_at": card.updated_at.isoformat() if hasattr(card, 'updated_at') and card.updated_at else None
            }
            
            return {
                "updated_card": updated_card_dict,
                "session_complete": False,  # Let frontend handle session completion
                "next_card": None  # TODO: Get next card if available
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Submit review error: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to submit review")

    @app.get("/api/fsrs/cards/due")
    async def get_due_cards(
        content_type: str,
        limit: Optional[int] = None,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Get cards due for review"""
        now = datetime.utcnow()
        
        query = db.query(FSRSCard).filter(
            FSRSCard.user_id == current_user.id,
            FSRSCard.content_type == content_type,
            FSRSCard.due_date <= now
        ).order_by(FSRSCard.due_date)
        
        if limit:
            query = query.limit(limit)
        
        cards = query.all()
        return cards

    @app.get("/api/fsrs/stats")
    async def get_study_stats(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Get comprehensive study statistics"""
        now = datetime.utcnow()
        
        # Overall stats
        total_cards = db.query(FSRSCard).filter_by(user_id=current_user.id).count()
        due_cards = db.query(FSRSCard).filter(
            FSRSCard.user_id == current_user.id,
            FSRSCard.due_date <= now
        ).count()
        
        overall_stats = {
            "total_cards": total_cards,
            "due_cards": due_cards,
            "retention_rate": 0.9,
            "study_streak": 0
        }
        
        vocabulary_stats = {
            "content_type": "vocabulary",
            "total_cards": db.query(FSRSCard).filter(
                FSRSCard.user_id == current_user.id,
                FSRSCard.content_type == "vocabulary"
            ).count()
        }
        
        grammar_stats = {
            "content_type": "grammar", 
            "total_cards": db.query(FSRSCard).filter(
                FSRSCard.user_id == current_user.id,
                FSRSCard.content_type == "grammar"
            ).count()
        }
        
        return {
            "overall_stats": overall_stats,
            "vocabulary_stats": vocabulary_stats,
            "grammar_stats": grammar_stats,
            "recent_sessions": []
        }

    @app.get("/api/fsrs/preferences")
    async def get_preferences(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Get user's FSRS preferences"""
        prefs = db.query(FSRSPreferences).filter_by(user_id=current_user.id).first()
        
        if not prefs:
            # Create default preferences
            prefs = FSRSPreferences(user_id=current_user.id)
            db.add(prefs)
            db.commit()
            db.refresh(prefs)
        
        return prefs

    @app.put("/api/fsrs/preferences")
    async def update_preferences(
        preferences: dict,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Update user's FSRS preferences"""
        prefs = db.query(FSRSPreferences).filter_by(user_id=current_user.id).first()
        
        if not prefs:
            prefs = FSRSPreferences(user_id=current_user.id)
            db.add(prefs)
        
        # Update preferences
        for key, value in preferences.items():
            if hasattr(prefs, key):
                setattr(prefs, key, value)
        
        prefs.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(prefs)
        
        return prefs

    @app.get("/api/fsrs/stats/quick")
    async def get_quick_stats(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Get quick stats for dashboard widgets"""
        now = datetime.utcnow()
        
        due_today = db.query(FSRSCard).filter(
            FSRSCard.user_id == current_user.id,
            FSRSCard.due_date <= now
        ).count()
        
        new_available = db.query(FSRSCard).filter(
            FSRSCard.user_id == current_user.id,
            FSRSCard.state == "new"
        ).count()
        
        return {
            "due_today": due_today,
            "new_available": new_available,
            "study_streak": 0  # TODO: Calculate from session history
        }

    return {
        'FSRSCard': FSRSCard,
        'FSRSReview': FSRSReview, 
        'FSRSStudySession': FSRSStudySession,
        'FSRSPreferences': FSRSPreferences
    }