from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Float, Text, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime, timedelta, date
from jose import JWTError, jwt
from passlib.context import CryptContext
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging

load_dotenv()

app = FastAPI(title="Korean Learning Platform API", version="1.0.0")

security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JWT_SECRET = os.getenv("JWT_SECRET", "your-super-secret-jwt-key-change-this-in-production")
GEMINI_API_KEY = os.getenv("GEMINI_API")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./korean_learning.db")

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_premium = Column(Boolean, default=False)
    is_admin = Column(Boolean, default=False)
    current_streak = Column(Integer, default=0)
    longest_streak = Column(Integer, default=0)
    last_study_date = Column(DateTime, nullable=True)
    last_active_date = Column(DateTime, nullable=True)
    total_study_time = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Profile fields
    display_name = Column(String, nullable=True)
    bio = Column(Text, nullable=True)
    profile_picture_url = Column(String, nullable=True)
    phone_number = Column(String, nullable=True)
    birth_date = Column(DateTime, nullable=True)
    country = Column(String, nullable=True)
    city = Column(String, nullable=True)
    
    # Learning preferences
    learning_level = Column(String, default="beginner")  # beginner, intermediate, advanced
    daily_goal_minutes = Column(Integer, default=30)
    preferred_study_time = Column(String, nullable=True)  # morning, afternoon, evening
    notification_enabled = Column(Boolean, default=True)
    email_notifications = Column(Boolean, default=True)
    
    # Billing information
    billing_address = Column(Text, nullable=True)
    subscription_type = Column(String, default="free")  # free, monthly, yearly
    subscription_expires_at = Column(DateTime, nullable=True)
    
    progress = relationship("Progress", back_populates="user")
    conversations = relationship("Conversation", back_populates="user")
    payments = relationship("PaymentVerification", back_populates="user")

class Vocabulary(Base):
    __tablename__ = "vocabulary"
    
    id = Column(Integer, primary_key=True, index=True)
    korean = Column(String, nullable=False, index=True)
    english = Column(String, nullable=False)
    pronunciation = Column(String, nullable=True)
    examples = Column(Text, nullable=True)  # JSON string
    category = Column(String, default="general")
    difficulty_level = Column(String, default="beginner")  # Changed to string
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class Grammar(Base):
    __tablename__ = "grammar"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    category = Column(String, nullable=False)
    explanation = Column(Text, nullable=False)
    examples = Column(Text, nullable=False)
    difficulty_level = Column(String, default="beginner")
    created_at = Column(DateTime, default=datetime.utcnow)

class Category(Base):
    __tablename__ = "categories"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    type = Column(String, nullable=False)  # 'vocabulary' or 'grammar'
    color = Column(String, default="#3B82F6")
    order = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class Progress(Base):
    __tablename__ = "progress"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content_type = Column(String, nullable=False)
    content_id = Column(Integer, nullable=False)
    mastery_level = Column(Integer, default=0)
    last_reviewed = Column(DateTime, default=datetime.utcnow)
    review_count = Column(Integer, default=0)
    
    user = relationship("User", back_populates="progress")

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)
    is_active = Column(Boolean, default=True, index=True)
    message_count = Column(Integer, default=0)  # Cache for quick stats
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    # Composite indexes for performance
    __table_args__ = (
        Index('idx_user_active_updated', 'user_id', 'is_active', 'updated_at'),
        Index('idx_user_created', 'user_id', 'created_at'),
    )

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False, index=True)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Performance optimizations
    content_length = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    # Composite indexes for common query patterns
    __table_args__ = (
        Index('idx_conversation_created', 'conversation_id', 'created_at'),
        Index('idx_conversation_role', 'conversation_id', 'role'),
    )


class PaymentVerification(Base):
    __tablename__ = "payment_verifications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    transaction_id = Column(String, nullable=False, unique=True)
    amount = Column(Float, nullable=False)
    status = Column(String, default="pending")
    verified_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="payments")

class DailyChallenge(Base):
    __tablename__ = "daily_challenges"
    
    id = Column(Integer, primary_key=True, index=True)
    challenge_date = Column(DateTime, nullable=False, index=True)
    vocabulary_words = Column(Text, nullable=False)  # JSON string of vocab IDs
    grammar_question_id = Column(Integer, ForeignKey("grammar.id"), nullable=False)
    ai_chat_prompt = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    grammar_question = relationship("Grammar")

class UserDailyChallengeProgress(Base):
    __tablename__ = "user_daily_challenge_progress"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    challenge_id = Column(Integer, ForeignKey("daily_challenges.id"), nullable=False)
    completed_vocabulary = Column(Text, default="[]")  # JSON string of completed vocab IDs
    completed_grammar = Column(Boolean, default=False)
    completed_chat = Column(Boolean, default=False)
    is_completed = Column(Boolean, default=False)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User")
    challenge = relationship("DailyChallenge")

Base.metadata.create_all(bind=engine)

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    is_premium: bool
    is_admin: bool
    current_streak: int
    longest_streak: int
    total_study_time: int
    created_at: datetime
    
    # Profile fields
    display_name: Optional[str] = None
    bio: Optional[str] = None
    profile_picture_url: Optional[str] = None
    phone_number: Optional[str] = None
    birth_date: Optional[datetime] = None
    country: Optional[str] = None
    city: Optional[str] = None
    
    # Learning preferences
    learning_level: str = "beginner"
    daily_goal_minutes: int = 30
    preferred_study_time: Optional[str] = None
    notification_enabled: bool = True
    email_notifications: bool = True
    
    # Billing information
    billing_address: Optional[str] = None
    subscription_type: str = "free"
    subscription_expires_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class UserProfileUpdate(BaseModel):
    display_name: Optional[str] = None
    bio: Optional[str] = None
    phone_number: Optional[str] = None
    birth_date: Optional[datetime] = None
    country: Optional[str] = None
    city: Optional[str] = None

class UserPreferencesUpdate(BaseModel):
    learning_level: Optional[str] = None
    daily_goal_minutes: Optional[int] = None
    preferred_study_time: Optional[str] = None
    notification_enabled: Optional[bool] = None
    email_notifications: Optional[bool] = None

class UserBillingUpdate(BaseModel):
    billing_address: Optional[str] = None

class UserPasswordChange(BaseModel):
    current_password: str
    new_password: str
    confirm_password: str

class VocabularyResponse(BaseModel):
    id: int
    korean: str
    english: str
    pronunciation: Optional[str] = None
    examples: List[str] = []
    category: str
    difficulty_level: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class VocabularyListResponse(BaseModel):
    items: List[VocabularyResponse]
    total: int
    skip: int
    limit: int

class GrammarResponse(BaseModel):
    id: int
    title: str
    category: str
    explanation: str
    examples: List[str]
    difficulty_level: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class VocabularyCreate(BaseModel):
    korean: str
    english: str
    pronunciation: Optional[str] = None
    examples: List[str] = []
    category: str
    difficulty_level: str

class VocabularyUpdate(BaseModel):
    korean: Optional[str] = None
    english: Optional[str] = None
    pronunciation: Optional[str] = None
    examples: Optional[List[str]] = None
    category: Optional[str] = None
    difficulty_level: Optional[str] = None

class GrammarCreate(BaseModel):
    title: str
    explanation: str
    examples: List[str]
    category: str
    difficulty_level: str

class GrammarUpdate(BaseModel):
    title: Optional[str] = None
    explanation: Optional[str] = None
    examples: Optional[List[str]] = None
    category: Optional[str] = None
    difficulty_level: Optional[str] = None

class CategoryResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    type: str
    color: str
    order: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class CategoryCreate(BaseModel):
    name: str
    description: Optional[str] = None
    type: str
    color: str = "#3B82F6"
    order: int = 0

class CategoryUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None
    color: Optional[str] = None
    order: Optional[int] = None

class ProgressCreate(BaseModel):
    content_type: str
    content_id: int
    mastery_level: int

class ProgressResponse(BaseModel):
    id: int
    content_type: str
    content_id: int
    mastery_level: int
    last_reviewed: datetime
    review_count: int
    
    class Config:
        from_attributes = True

class StudySessionCreate(BaseModel):
    study_time_minutes: int

# New chat models
class ConversationResponse(BaseModel):
    id: int
    user_id: int
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    is_active: bool
    message_count: int
    
    class Config:
        from_attributes = True

class MessageResponse(BaseModel):
    id: int
    conversation_id: int
    role: str
    content: str
    created_at: datetime
    content_length: Optional[int] = None
    token_count: Optional[int] = None
    
    class Config:
        from_attributes = True

class ConversationWithMessagesResponse(BaseModel):
    id: int
    user_id: int
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    is_active: bool
    message_count: int
    messages: List[MessageResponse] = []
    
    class Config:
        from_attributes = True

class MessageCreate(BaseModel):
    content: str

class ConversationCreate(BaseModel):
    title: Optional[str] = None
    initial_message: Optional[str] = None

class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    is_active: Optional[bool] = None

# Legacy chat models (for backward compatibility)
class ChatMessageCreate(BaseModel):
    message: str

class ChatResponse(BaseModel):
    id: int
    message: str
    response: str
    timestamp: str

class ChatMessageResponse(BaseModel):
    id: int
    user_id: int
    message: str
    response: str
    timestamp: str
    created_at: str

class ChatHistoryResponse(BaseModel):
    messages: List[ChatMessageResponse]
    total: int
    limit: int

class PaymentSubmit(BaseModel):
    transaction_id: str
    amount: float

class PaymentResponse(BaseModel):
    id: int
    transaction_id: str
    amount: float
    status: str
    verified_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class AuthResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class UserDailyChallengeProgressResponse(BaseModel):
    id: int
    completed_vocabulary: List[int]
    completed_grammar: bool
    completed_chat: bool
    is_completed: bool
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class DailyChallengeResponse(BaseModel):
    id: int
    challenge_date: str
    vocabulary_words: List[VocabularyResponse]
    grammar_question: GrammarResponse
    ai_chat_prompt: str
    progress: Optional[UserDailyChallengeProgressResponse] = None
    
    class Config:
        from_attributes = True

class DailyChallengeProgressUpdate(BaseModel):
    vocabulary_id: Optional[int] = None
    grammar_completed: Optional[bool] = None
    chat_completed: Optional[bool] = None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=30)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm="HS256")
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=["HS256"])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return email
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user(email: str = Depends(verify_token), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

def get_admin_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

def update_streak(user: User, db: Session, activity_type: str = "study"):
    today = date.today()
    
    # Use appropriate date field based on activity type
    if activity_type == "login":
        last_activity_date = user.last_active_date
    else:
        last_activity_date = user.last_study_date
    
    if last_activity_date:
        last_activity = last_activity_date.date()
        days_diff = (today - last_activity).days
        
        if days_diff == 1:
            user.current_streak += 1
        elif days_diff > 1:
            user.current_streak = 1
        # if days_diff == 0, streak stays the same (same day activity)
    else:
        user.current_streak = 1
    
    if user.current_streak > user.longest_streak:
        user.longest_streak = user.current_streak
    
    # Update appropriate date field
    current_time = datetime.utcnow()
    if activity_type == "login":
        user.last_active_date = current_time
    else:
        user.last_study_date = current_time
    
    db.commit()

def generate_daily_challenge(target_date: date, db: Session):
    """Generate a daily challenge for a specific date"""
    import json
    import random
    
    # Check if challenge already exists for this date
    existing_challenge = db.query(DailyChallenge).filter(
        DailyChallenge.challenge_date == target_date
    ).first()
    
    if existing_challenge:
        return existing_challenge
    
    # Get 5 random vocabulary words
    vocab_count = db.query(Vocabulary).count()
    if vocab_count < 5:
        # If not enough vocabulary, get all available
        vocabulary_words = db.query(Vocabulary).all()
    else:
        # Get random vocabulary words
        random_offsets = random.sample(range(vocab_count), min(5, vocab_count))
        vocabulary_words = []
        for offset in random_offsets:
            vocab = db.query(Vocabulary).offset(offset).first()
            if vocab:
                vocabulary_words.append(vocab)
    
    # Get 1 random grammar question
    grammar_count = db.query(Grammar).count()
    if grammar_count == 0:
        return None  # No grammar questions available
    
    random_grammar_offset = random.randint(0, grammar_count - 1)
    grammar_question = db.query(Grammar).offset(random_grammar_offset).first()
    
    # AI chat prompts pool
    chat_prompts = [
        "Practice introducing yourself in Korean to a new friend.",
        "Describe your favorite Korean food and explain why you like it.",
        "Ask about directions to a popular tourist spot in Seoul.",
        "Practice ordering food at a Korean restaurant.",
        "Describe your daily routine using Korean time expressions.",
        "Talk about your hobbies and interests in Korean.",
        "Practice making plans with a Korean friend for the weekend.",
        "Describe the weather today and what clothes you're wearing.",
        "Talk about your family members and their jobs.",
        "Practice asking for help when you're lost in Korea."
    ]
    
    # Select random chat prompt
    ai_chat_prompt = random.choice(chat_prompts)
    
    # Create the challenge
    challenge = DailyChallenge(
        challenge_date=target_date,
        vocabulary_words=json.dumps([vocab.id for vocab in vocabulary_words]),
        grammar_question_id=grammar_question.id,
        ai_chat_prompt=ai_chat_prompt,
        created_at=datetime.utcnow()
    )
    
    db.add(challenge)
    db.commit()
    db.refresh(challenge)
    
    return challenge

def get_or_create_daily_challenge(target_date: date, db: Session):
    """Get existing challenge or create new one for the date"""
    challenge = db.query(DailyChallenge).filter(
        DailyChallenge.challenge_date == target_date
    ).first()
    
    if not challenge:
        challenge = generate_daily_challenge(target_date, db)
    
    return challenge

@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "Korean Learning Platform API"}

@app.get("/test-auth-response")
async def test_auth_response():
    return {
        "access_token": "test_token",
        "token_type": "bearer",
        "user": {
            "id": 1,
            "email": "test@example.com",
            "is_premium": False,
            "current_streak": 0,
            "longest_streak": 0,
            "total_study_time": 0
        }
    }

@app.post("/api/auth/register")
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    try:
        db_user = db.query(User).filter(User.email == user.email).first()
        if db_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        hashed_password = get_password_hash(user.password)
        db_user = User(email=user.email, hashed_password=hashed_password)
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        access_token = create_access_token(data={"sub": user.email})
        logger.info(f"User registered: {user.email}")
        
        user_response = {
            "id": db_user.id,
            "email": db_user.email,
            "is_premium": db_user.is_premium,
            "is_admin": db_user.is_admin,
            "current_streak": db_user.current_streak,
            "longest_streak": db_user.longest_streak,
            "total_study_time": db_user.total_study_time
        }
        
        return {"access_token": access_token, "token_type": "bearer", "user": user_response}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/api/auth/login")
def login_user(user: UserLogin, db: Session = Depends(get_db)):
    try:
        db_user = db.query(User).filter(User.email == user.email).first()
        if not db_user or not verify_password(user.password, db_user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Update login streak
        update_streak(db_user, db, activity_type="login")
        
        access_token = create_access_token(data={"sub": user.email})
        logger.info(f"User logged in: {user.email}")
        
        user_response = {
            "id": db_user.id,
            "email": db_user.email,
            "is_premium": db_user.is_premium,
            "is_admin": db_user.is_admin,
            "current_streak": db_user.current_streak,
            "longest_streak": db_user.longest_streak,
            "total_study_time": db_user.total_study_time
        }
        
        response_data = {"access_token": access_token, "token_type": "bearer", "user": user_response}
        return response_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.get("/api/user/profile", response_model=UserResponse)
def get_user_profile(current_user: User = Depends(get_current_user)):
    return current_user

@app.get("/api/auth/me", response_model=UserResponse)
def get_current_user_profile(current_user: User = Depends(get_current_user)):
    return current_user

# Profile Management Endpoints
@app.put("/api/user/profile", response_model=UserResponse)
def update_user_profile(profile_data: UserProfileUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        # Update profile fields
        for field, value in profile_data.model_dump(exclude_unset=True).items():
            setattr(current_user, field, value)
        
        db.commit()
        db.refresh(current_user)
        return current_user
    except Exception as e:
        db.rollback()
        logger.error(f"Profile update error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update profile")

@app.put("/api/user/preferences", response_model=UserResponse)
def update_user_preferences(preferences_data: UserPreferencesUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        # Update preferences fields
        for field, value in preferences_data.model_dump(exclude_unset=True).items():
            setattr(current_user, field, value)
        
        db.commit()
        db.refresh(current_user)
        return current_user
    except Exception as e:
        db.rollback()
        logger.error(f"Preferences update error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update preferences")

@app.put("/api/user/billing", response_model=UserResponse)
def update_user_billing(billing_data: UserBillingUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        # Update billing fields
        for field, value in billing_data.model_dump(exclude_unset=True).items():
            setattr(current_user, field, value)
        
        db.commit()
        db.refresh(current_user)
        return current_user
    except Exception as e:
        db.rollback()
        logger.error(f"Billing update error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update billing information")

@app.post("/api/user/change-password")
def change_user_password(password_data: UserPasswordChange, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        # Verify current password
        if not pwd_context.verify(password_data.current_password, current_user.hashed_password):
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        
        # Verify new password confirmation
        if password_data.new_password != password_data.confirm_password:
            raise HTTPException(status_code=400, detail="New passwords do not match")
        
        # Update password
        current_user.hashed_password = pwd_context.hash(password_data.new_password)
        db.commit()
        
        return {"message": "Password updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Password change error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to change password")

@app.post("/api/user/upload-avatar")
async def upload_user_avatar(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Placeholder for file upload functionality
    # In a real implementation, you would handle file upload here
    return {"message": "Avatar upload endpoint - implementation pending"}

@app.delete("/api/user/account")
def delete_user_account(password: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        # Verify password before deletion
        if not pwd_context.verify(password, current_user.hashed_password):
            raise HTTPException(status_code=400, detail="Password is incorrect")
        
        # In a real implementation, you would:
        # 1. Delete all user data (progress, chat history, etc.)
        # 2. Cancel subscriptions
        # 3. Send confirmation email
        # For now, just mark as deleted or actually delete
        
        # Delete user and all related data
        db.delete(current_user)
        db.commit()
        
        return {"message": "Account deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Account deletion error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete account")

@app.get("/api/user/export")
def export_user_data(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        # Collect all user data
        user_data = {
            "profile": {
                "id": current_user.id,
                "email": current_user.email,
                "display_name": current_user.display_name,
                "bio": current_user.bio,
                "country": current_user.country,
                "city": current_user.city,
                "learning_level": current_user.learning_level,
                "created_at": current_user.created_at.isoformat() if current_user.created_at else None
            },
            "learning_stats": {
                "current_streak": current_user.current_streak,
                "longest_streak": current_user.longest_streak,
                "total_study_time": current_user.total_study_time,
                "daily_goal_minutes": current_user.daily_goal_minutes
            },
            "progress": [],
            "chat_history": []
        }
        
        # Get user progress
        progress_records = db.query(Progress).filter(Progress.user_id == current_user.id).all()
        for progress in progress_records:
            user_data["progress"].append({
                "content_type": progress.content_type,
                "content_id": progress.content_id,
                "mastery_level": progress.mastery_level,
                "last_reviewed": progress.last_reviewed.isoformat() if progress.last_reviewed else None,
                "review_count": progress.review_count
            })
        
        # Get chat history
        chat_records = db.query(ChatConversation).filter(ChatConversation.user_id == current_user.id).all()
        for chat in chat_records:
            user_data["chat_history"].append({
                "message": chat.message,
                "response": chat.response,
                "timestamp": chat.timestamp.isoformat() if chat.timestamp else None
            })
        
        return user_data
    except Exception as e:
        logger.error(f"Data export error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to export user data")

@app.get("/api/vocabulary", response_model=VocabularyListResponse)
def get_vocabulary(skip: int = 0, limit: int = 20, search: Optional[str] = None, category: Optional[str] = None, difficulty_level: Optional[str] = None, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    import json
    
    query = db.query(Vocabulary)
    
    # Apply filters
    if search:
        query = query.filter(
            Vocabulary.korean.contains(search) | 
            Vocabulary.english.contains(search)
        )
    if category:
        query = query.filter(Vocabulary.category == category)
    if difficulty_level:
        query = query.filter(Vocabulary.difficulty_level == difficulty_level)
    
    # Get total count for pagination
    total = query.count()
    
    # Get paginated results
    vocabulary_items = query.offset(skip).limit(limit).all()
    
    # Process vocabulary items to handle examples JSON field
    processed_items = []
    for item in vocabulary_items:
        item_dict = {
            "id": item.id,
            "korean": item.korean,
            "english": item.english,
            "pronunciation": item.pronunciation,
            "examples": json.loads(item.examples) if item.examples else [],
            "category": item.category,
            "difficulty_level": item.difficulty_level,
            "created_at": item.created_at,
            "updated_at": item.updated_at or item.created_at
        }
        processed_items.append(VocabularyResponse(**item_dict))
    
    return VocabularyListResponse(
        items=processed_items,
        total=total,
        skip=skip,
        limit=limit
    )

@app.get("/api/vocabulary/categories")
def get_vocabulary_categories(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        # Get distinct categories from vocabulary table along with counts
        from sqlalchemy import func, distinct
        
        # Get category stats from vocabulary table
        category_stats = db.query(
            Vocabulary.category,
            func.count(Vocabulary.id).label('total_count')
        ).group_by(Vocabulary.category).all()
        
        # Get user progress for vocabulary items
        progress_stats = db.query(
            Vocabulary.category,
            func.count(Progress.id).label('completed_count')
        ).join(
            Progress, 
            (Progress.content_type == 'vocabulary') & (Progress.content_id == Vocabulary.id) & (Progress.mastery_level >= 100) & (Progress.user_id == current_user.id)
        ).group_by(Vocabulary.category).all()
        
        # Create a dictionary for quick lookup
        progress_dict = {stat.category: stat.completed_count for stat in progress_stats}
        
        # Build category response with mock data structure
        categories = []
        # Define gradients for different difficulty levels
        gradient_colors = {
            'beginner': 'linear-gradient(135deg, #006eff 0%, #0072ff 100%)',      # Blue gradient
            'intermediate': 'linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%)', # Purple gradient
            'advanced': 'linear-gradient(135deg, #dc2626 0%, #ef4444 100%)'      # Red gradient
        }
        
        category_info = {
            'basic': {
                'name': 'Basic Words',
                'description': 'Essential Korean words for beginners. Start your Korean journey here!',
                'color': gradient_colors['beginner'],
                'icon': '🌟',
                'difficulty_level': 'beginner'
            },
            'food': {
                'name': 'Food & Drinks',
                'description': 'Learn vocabulary related to Korean cuisine, dining, and beverages.',
                'color': gradient_colors['beginner'],
                'icon': '🍜',
                'difficulty_level': 'beginner'
            },
            'family': {
                'name': 'Family & Relations',
                'description': 'Words for family members, relationships, and social connections.',
                'color': gradient_colors['beginner'],
                'icon': '👨‍👩‍👧‍👦',
                'difficulty_level': 'beginner'
            },
            'colors': {
                'name': 'Colors & Shapes',
                'description': 'Learn colors, shapes, and descriptive words for visual elements.',
                'color': gradient_colors['beginner'],
                'icon': '🎨',
                'difficulty_level': 'beginner'
            },
            'numbers': {
                'name': 'Numbers & Counting',
                'description': 'Master Korean numbers, counting systems, and mathematical terms.',
                'color': gradient_colors['intermediate'],
                'icon': '🔢',
                'difficulty_level': 'intermediate'
            },
            'time': {
                'name': 'Time & Dates',
                'description': 'Vocabulary for time, dates, seasons, and temporal expressions.',
                'color': gradient_colors['intermediate'],
                'icon': '⏰',
                'difficulty_level': 'intermediate'
            },
            'travel': {
                'name': 'Travel & Transportation',
                'description': 'Essential words for traveling, transportation, and navigation in Korea.',
                'color': gradient_colors['intermediate'],
                'icon': '✈️',
                'difficulty_level': 'intermediate'
            },
            'business': {
                'name': 'Business & Work',
                'description': 'Professional vocabulary for workplace communication and business.',
                'color': gradient_colors['advanced'],
                'icon': '💼',
                'difficulty_level': 'advanced'
            }
        }
        
        # Define learning level hierarchy for progressive access
        level_hierarchy = {
            'beginner': ['beginner'],
            'intermediate': ['beginner', 'intermediate'],
            'advanced': ['beginner', 'intermediate', 'advanced']
        }
        
        # Get user's accessible difficulty levels
        user_level = current_user.learning_level or 'beginner'
        accessible_levels = level_hierarchy.get(user_level, ['beginner'])
        
        for stat in category_stats:
            category_id = stat.category
            total_count = stat.total_count
            completed_count = progress_dict.get(category_id, 0)
            
            # Get category info or use default
            # Determine difficulty level based on actual vocabulary data
            vocab_difficulties = db.query(Vocabulary.difficulty_level).filter(Vocabulary.category == category_id).all()
            max_difficulty_num = max([int(d.difficulty_level) if str(d.difficulty_level).isdigit() else 1 for d in vocab_difficulties], default=1)
            
            # Map numeric difficulty to string
            difficulty_mapping = {1: 'beginner', 2: 'intermediate', 3: 'advanced'}
            actual_difficulty = difficulty_mapping.get(max_difficulty_num, 'beginner')
            
            info = category_info.get(category_id, {
                'name': category_id.title(),
                'description': f'Learn {category_id} vocabulary',
                'color': gradient_colors[actual_difficulty],
                'icon': '📖',
                'difficulty_level': actual_difficulty
            })
            
            # Override difficulty level if not in predefined categories
            if category_id not in category_info:
                info['difficulty_level'] = actual_difficulty
                info['color'] = gradient_colors[actual_difficulty]
            
            # Only include categories that match user's learning level or below
            if info['difficulty_level'] in accessible_levels:
                categories.append({
                    'id': category_id,
                    'name': info['name'],
                    'description': info['description'],
                    'color': info['color'],
                    'icon': info['icon'],
                    'word_count': total_count,
                    'completed_count': completed_count,
                    'difficulty_level': info['difficulty_level']
                })
        
        return categories
    except Exception as e:
        logger.error(f"Vocabulary categories fetch error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch vocabulary categories")

@app.get("/api/vocabulary/categories/{category_id}/stats")
def get_category_stats(category_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        from sqlalchemy import func
        
        # Get total count for this category
        total_count = db.query(func.count(Vocabulary.id)).filter(Vocabulary.category == category_id).scalar()
        
        # Get completed count for this user
        completed_count = db.query(func.count(Progress.id)).filter(
            Progress.content_type == 'vocabulary',
            Progress.user_id == current_user.id,
            Progress.mastery_level >= 100
        ).join(
            Vocabulary, Progress.content_id == Vocabulary.id
        ).filter(Vocabulary.category == category_id).scalar()
        
        # Get in-progress count
        in_progress_count = db.query(func.count(Progress.id)).filter(
            Progress.content_type == 'vocabulary',
            Progress.user_id == current_user.id,
            Progress.mastery_level > 0,
            Progress.mastery_level < 100
        ).join(
            Vocabulary, Progress.content_id == Vocabulary.id
        ).filter(Vocabulary.category == category_id).scalar()
        
        return {
            'total_count': total_count or 0,
            'completed_count': completed_count or 0,
            'in_progress_count': in_progress_count or 0,
            'progress_percentage': round((completed_count / total_count * 100) if total_count > 0 else 0, 1)
        }
    except Exception as e:
        logger.error(f"Category stats fetch error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch category stats")

@app.get("/api/vocabulary/{vocabulary_id}", response_model=VocabularyResponse)
def get_vocabulary_item(vocabulary_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    import json
    
    vocabulary = db.query(Vocabulary).filter(Vocabulary.id == vocabulary_id).first()
    if not vocabulary:
        raise HTTPException(status_code=404, detail="Vocabulary item not found")
    
    # Process the vocabulary item to handle examples JSON field
    item_dict = {
        "id": vocabulary.id,
        "korean": vocabulary.korean,
        "english": vocabulary.english,
        "pronunciation": vocabulary.pronunciation,
        "examples": json.loads(vocabulary.examples) if vocabulary.examples else [],
        "category": vocabulary.category,
        "difficulty_level": vocabulary.difficulty_level,
        "created_at": vocabulary.created_at,
        "updated_at": vocabulary.updated_at or vocabulary.created_at
    }
    return VocabularyResponse(**item_dict)

@app.get("/api/grammar", response_model=List[GrammarResponse])
def get_grammar(category: Optional[str] = None, difficulty_level: Optional[str] = None, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    import json
    
    try:
        query = db.query(Grammar)
        if category:
            query = query.filter(Grammar.category == category)
        if difficulty_level:
            query = query.filter(Grammar.difficulty_level == difficulty_level)
        grammar_items = query.all()
        
        # Process grammar items to handle examples JSON field
        processed_items = []
        for item in grammar_items:
            try:
                # Handle examples field more safely
                examples = []
                if item.examples:
                    if isinstance(item.examples, str):
                        try:
                            examples = json.loads(item.examples)
                        except json.JSONDecodeError:
                            # If JSON parsing fails, treat as a single example
                            examples = [item.examples]
                    elif isinstance(item.examples, list):
                        examples = item.examples
                    else:
                        examples = [str(item.examples)]
                
                item_dict = {
                    "id": item.id,
                    "title": item.title,
                    "category": item.category,
                    "explanation": item.explanation,
                    "examples": examples,
                    "difficulty_level": item.difficulty_level,
                    "created_at": item.created_at,
                    "updated_at": item.created_at  # Use created_at since updated_at doesn't exist
                }
                processed_items.append(GrammarResponse(**item_dict))
            except Exception as e:
                logger.error(f"Error processing grammar item {item.id}: {str(e)}")
                continue
        
        return processed_items
    except Exception as e:
        logger.error(f"Grammar fetch error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch grammar: {str(e)}")

@app.get("/api/grammar/{grammar_id}", response_model=GrammarResponse)
def get_grammar_item(grammar_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    import json
    
    grammar = db.query(Grammar).filter(Grammar.id == grammar_id).first()
    if not grammar:
        raise HTTPException(status_code=404, detail="Grammar item not found")
    
    # Process the grammar item to handle examples JSON field
    try:
        examples = []
        if grammar.examples:
            if isinstance(grammar.examples, str):
                try:
                    examples = json.loads(grammar.examples)
                except json.JSONDecodeError:
                    examples = [grammar.examples]
            elif isinstance(grammar.examples, list):
                examples = grammar.examples
            else:
                examples = [str(grammar.examples)]
    except Exception:
        examples = []
        
    item_dict = {
        "id": grammar.id,
        "title": grammar.title,
        "category": grammar.category,
        "explanation": grammar.explanation,
        "examples": examples,
        "difficulty_level": grammar.difficulty_level,
        "created_at": grammar.created_at,
        "updated_at": grammar.created_at  # Use created_at since updated_at doesn't exist
    }
    return GrammarResponse(**item_dict)

@app.post("/api/progress", response_model=ProgressResponse)
def update_progress(progress: ProgressCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        existing_progress = db.query(Progress).filter(
            Progress.user_id == current_user.id,
            Progress.content_type == progress.content_type,
            Progress.content_id == progress.content_id
        ).first()
        
        if existing_progress:
            existing_progress.mastery_level = progress.mastery_level
            existing_progress.last_reviewed = datetime.utcnow()
            existing_progress.review_count += 1
            db.commit()
            db.refresh(existing_progress)
            return existing_progress
        else:
            new_progress = Progress(
                user_id=current_user.id,
                content_type=progress.content_type,
                content_id=progress.content_id,
                mastery_level=progress.mastery_level,
                review_count=1
            )
            db.add(new_progress)
            db.commit()
            db.refresh(new_progress)
            return new_progress
    except Exception as e:
        logger.error(f"Progress update error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update progress")

@app.get("/api/progress", response_model=List[ProgressResponse])
def get_progress(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    progress = db.query(Progress).filter(Progress.user_id == current_user.id).all()
    return progress

@app.post("/api/study-session")
def log_study_session(session: StudySessionCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        current_user.total_study_time += session.study_time_minutes
        update_streak(current_user, db, activity_type="study")
        
        logger.info(f"Study session logged for user {current_user.email}: {session.study_time_minutes} minutes")
        return {
            "message": "Study session logged successfully",
            "current_streak": current_user.current_streak,
            "total_study_time": current_user.total_study_time
        }
    except Exception as e:
        logger.error(f"Study session logging error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to log study session")

@app.get("/api/streak/milestones")
def get_streak_milestones(current_user: User = Depends(get_current_user)):
    """Get milestone badges based on current streak"""
    try:
        current_streak = current_user.current_streak
        longest_streak = current_user.longest_streak
        
        milestones = [
            {"days": 3, "emoji": "🔥", "name": "Fire Starter", "description": "3 day streak!"},
            {"days": 7, "emoji": "⭐", "name": "Week Warrior", "description": "7 day streak!"},
            {"days": 30, "emoji": "💎", "name": "Diamond Learner", "description": "30 day streak!"},
            {"days": 100, "emoji": "👑", "name": "Streak Master", "description": "100 day streak!"},
        ]
        
        achieved_milestones = []
        next_milestone = None
        
        for milestone in milestones:
            if current_streak >= milestone["days"]:
                achieved_milestones.append({**milestone, "achieved": True})
            elif next_milestone is None:
                next_milestone = {**milestone, "achieved": False, "days_remaining": milestone["days"] - current_streak}
        
        return {
            "current_streak": current_streak,
            "longest_streak": longest_streak,
            "achieved_milestones": achieved_milestones,
            "next_milestone": next_milestone,
            "all_milestones": milestones
        }
    except Exception as e:
        logger.error(f"Milestone fetch error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch milestones")

@app.get("/api/daily-challenge", response_model=DailyChallengeResponse)
def get_daily_challenge(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get today's daily challenge"""
    try:
        import json
        
        today = date.today()
        challenge = get_or_create_daily_challenge(today, db)
        
        if not challenge:
            raise HTTPException(status_code=404, detail="No challenge available")
        
        # Get vocabulary words
        vocab_ids = json.loads(challenge.vocabulary_words)
        vocabulary_words = []
        for vocab_id in vocab_ids:
            vocab = db.query(Vocabulary).filter(Vocabulary.id == vocab_id).first()
            if vocab:
                vocab_dict = {
                    "id": vocab.id,
                    "korean": vocab.korean,
                    "english": vocab.english,
                    "pronunciation": vocab.pronunciation,
                    "examples": json.loads(vocab.examples) if vocab.examples else [],
                    "category": vocab.category,
                    "difficulty_level": vocab.difficulty_level,
                    "created_at": vocab.created_at,
                    "updated_at": vocab.updated_at or vocab.created_at
                }
                vocabulary_words.append(VocabularyResponse(**vocab_dict))
        
        # Get grammar question
        grammar = challenge.grammar_question
        grammar_examples = []
        if grammar.examples:
            if isinstance(grammar.examples, str):
                try:
                    grammar_examples = json.loads(grammar.examples)
                except json.JSONDecodeError:
                    grammar_examples = [grammar.examples]
            elif isinstance(grammar.examples, list):
                grammar_examples = grammar.examples
            else:
                grammar_examples = [str(grammar.examples)]
        
        grammar_dict = {
            "id": grammar.id,
            "title": grammar.title,
            "category": grammar.category,
            "explanation": grammar.explanation,
            "examples": grammar_examples,
            "difficulty_level": grammar.difficulty_level,
            "created_at": grammar.created_at,
            "updated_at": grammar.created_at
        }
        grammar_response = GrammarResponse(**grammar_dict)
        
        # Get user's progress
        progress = db.query(UserDailyChallengeProgress).filter(
            UserDailyChallengeProgress.user_id == current_user.id,
            UserDailyChallengeProgress.challenge_id == challenge.id
        ).first()
        
        progress_response = None
        if progress:
            completed_vocab = json.loads(progress.completed_vocabulary) if progress.completed_vocabulary else []
            progress_response = UserDailyChallengeProgressResponse(
                id=progress.id,
                completed_vocabulary=completed_vocab,
                completed_grammar=progress.completed_grammar,
                completed_chat=progress.completed_chat,
                is_completed=progress.is_completed,
                completed_at=progress.completed_at
            )
        
        return DailyChallengeResponse(
            id=challenge.id,
            challenge_date=challenge.challenge_date.isoformat(),
            vocabulary_words=vocabulary_words,
            grammar_question=grammar_response,
            ai_chat_prompt=challenge.ai_chat_prompt,
            progress=progress_response
        )
        
    except Exception as e:
        logger.error(f"Daily challenge fetch error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch daily challenge")

@app.post("/api/daily-challenge/progress")
def update_daily_challenge_progress(
    progress_update: DailyChallengeProgressUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user's progress on today's daily challenge"""
    try:
        import json
        
        today = date.today()
        challenge = get_or_create_daily_challenge(today, db)
        
        if not challenge:
            raise HTTPException(status_code=404, detail="No challenge available")
        
        # Get or create progress record
        progress = db.query(UserDailyChallengeProgress).filter(
            UserDailyChallengeProgress.user_id == current_user.id,
            UserDailyChallengeProgress.challenge_id == challenge.id
        ).first()
        
        if not progress:
            progress = UserDailyChallengeProgress(
                user_id=current_user.id,
                challenge_id=challenge.id,
                completed_vocabulary="[]",
                completed_grammar=False,
                completed_chat=False,
                is_completed=False
            )
            db.add(progress)
            db.flush()  # Flush to get the ID
        
        # Update progress based on the request
        if progress_update.vocabulary_id is not None:
            completed_vocab = json.loads(progress.completed_vocabulary) if progress.completed_vocabulary else []
            if progress_update.vocabulary_id not in completed_vocab:
                completed_vocab.append(progress_update.vocabulary_id)
                progress.completed_vocabulary = json.dumps(completed_vocab)
        
        if progress_update.grammar_completed is not None:
            progress.completed_grammar = progress_update.grammar_completed
        
        if progress_update.chat_completed is not None:
            progress.completed_chat = progress_update.chat_completed
        
        # Check if challenge is fully completed
        vocab_ids = json.loads(challenge.vocabulary_words)
        completed_vocab = json.loads(progress.completed_vocabulary) if progress.completed_vocabulary else []
        
        vocab_complete = len(completed_vocab) >= len(vocab_ids)
        grammar_complete = progress.completed_grammar
        chat_complete = progress.completed_chat
        
        if vocab_complete and grammar_complete and chat_complete and not progress.is_completed:
            progress.is_completed = True
            progress.completed_at = datetime.utcnow()
            
            # Update user streak for challenge completion
            update_streak(current_user, db, activity_type="study")
        
        db.commit()
        
        return {
            "message": "Progress updated successfully",
            "is_completed": progress.is_completed,
            "completed_vocabulary": len(completed_vocab),
            "total_vocabulary": len(vocab_ids),
            "completed_grammar": progress.completed_grammar,
            "completed_chat": progress.completed_chat
        }
        
    except Exception as e:
        logger.error(f"Daily challenge progress update error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update progress")

# New optimized chat endpoints
@app.get("/api/conversations", response_model=List[ConversationResponse])
def get_conversations(
    skip: int = 0, 
    limit: int = 20,
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    """Get user's conversations with pagination"""
    conversations = db.query(Conversation)\
        .filter(Conversation.user_id == current_user.id, Conversation.is_active == True)\
        .order_by(Conversation.updated_at.desc())\
        .offset(skip).limit(limit).all()
    return conversations

@app.post("/api/conversations", response_model=ConversationResponse)
def create_conversation(
    request: ConversationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new conversation"""
    conversation = Conversation(
        user_id=current_user.id,
        title=request.title or f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    
    # If initial message provided, add it
    if request.initial_message:
        message = Message(
            conversation_id=conversation.id,
            role="user",
            content=request.initial_message,
            created_at=datetime.utcnow(),
            content_length=len(request.initial_message)
        )
        db.add(message)
        conversation.message_count = 1
        conversation.updated_at = datetime.utcnow()
        db.commit()
    
    return conversation

@app.get("/api/conversations/{conversation_id}", response_model=ConversationWithMessagesResponse)
def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific conversation with its messages"""
    conversation = db.query(Conversation)\
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)\
        .first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    messages = db.query(Message)\
        .filter(Message.conversation_id == conversation_id)\
        .order_by(Message.created_at.asc()).all()
    
    return ConversationWithMessagesResponse(
        id=conversation.id,
        user_id=conversation.user_id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        is_active=conversation.is_active,
        message_count=conversation.message_count,
        messages=[MessageResponse(
            id=msg.id,
            conversation_id=msg.conversation_id,
            role=msg.role,
            content=msg.content,
            created_at=msg.created_at,
            content_length=msg.content_length,
            token_count=msg.token_count
        ) for msg in messages]
    )

@app.post("/api/conversations/{conversation_id}/messages", response_model=MessageResponse)
def send_message(
    conversation_id: int,
    request: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Send a message in a conversation and get AI response"""
    try:
        # Verify conversation exists and belongs to user
        conversation = db.query(Conversation)\
            .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)\
            .first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="AI service not configured")
        
        # Save user message
        user_message = Message(
            conversation_id=conversation_id,
            role="user",
            content=request.content,
            created_at=datetime.utcnow(),
            content_length=len(request.content)
        )
        db.add(user_message)
        
        # Get recent messages for context (last 10 messages)
        recent_messages = db.query(Message)\
            .filter(Message.conversation_id == conversation_id)\
            .order_by(Message.created_at.desc())\
            .limit(10).all()
        
        # Build conversation context for Gemini
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        system_prompt = """You are a helpful Korean language tutor named 선생님 (Teacher). Help users learn Korean vocabulary, grammar, pronunciation, and culture. 

        Key guidelines:
        1. Always be encouraging and patient
        2. For Korean words/phrases, provide: Korean text (한글), romanization, English meaning, example sentences
        3. Explain grammar concepts clearly with examples
        4. Share cultural context when relevant
        5. Adapt to the user's learning level
        6. Use both Korean and English appropriately
        7. Keep responses concise but informative
        8. If the user speaks Korean, praise their effort and gently correct if needed

        Response format:
        - Use emojis sparingly (📚 for learning, 👍 for encouragement)
        - Structure your responses clearly
        - Provide practical examples
        - End with encouragement or a follow-up question when appropriate"""
        
        # Build conversation history
        conversation_history = ""
        for msg in reversed(recent_messages):
            role_name = "Teacher" if msg.role == "assistant" else "User"
            conversation_history += f"{role_name}: {msg.content}\n\n"
        
        # Create the full prompt with context
        full_prompt = f"""{system_prompt}

Previous conversation:
{conversation_history}

Current message: {request.content}

Please respond as the Korean tutor:"""
        
        # Get AI response
        response = model.generate_content(full_prompt)
        ai_response_content = response.text
        
        # Save AI response
        ai_message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=ai_response_content,
            created_at=datetime.utcnow(),
            content_length=len(ai_response_content)
        )
        db.add(ai_message)
        
        # Update conversation metadata
        conversation.message_count += 2  # user + assistant
        conversation.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(ai_message)
        
        return MessageResponse(
            id=ai_message.id,
            conversation_id=ai_message.conversation_id,
            role=ai_message.role,
            content=ai_message.content,
            created_at=ai_message.created_at,
            content_length=ai_message.content_length,
            token_count=ai_message.token_count
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")

@app.put("/api/conversations/{conversation_id}", response_model=ConversationResponse)
def update_conversation(
    conversation_id: int,
    request: ConversationUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update conversation (title, active status, etc.)"""
    conversation = db.query(Conversation)\
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)\
        .first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if request.title is not None:
        conversation.title = request.title
    if request.is_active is not None:
        conversation.is_active = request.is_active
    
    conversation.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(conversation)
    
    return conversation

@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Soft delete a conversation"""
    conversation = db.query(Conversation)\
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)\
        .first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation.is_active = False
    conversation.updated_at = datetime.utcnow()
    db.commit()
    
    return {"message": "Conversation deleted successfully"}

# Legacy chat endpoints (for backward compatibility)
@app.post("/api/chat", response_model=ChatResponse)
def chat_with_ai(request: ChatMessageCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Legacy chat endpoint - creates a new conversation or uses existing one"""
    try:
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="AI service not configured")
        
        # Get or create a default conversation for legacy support
        today = datetime.utcnow().date()
        conversation = db.query(Conversation)\
            .filter(
                Conversation.user_id == current_user.id,
                Conversation.is_active == True,
                Conversation.created_at >= today
            ).first()
        
        if not conversation:
            conversation = Conversation(
                user_id=current_user.id,
                title=f"Chat {today.strftime('%Y-%m-%d')}",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
        
        # Use the new message endpoint logic
        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=request.message,
            created_at=datetime.utcnow(),
            content_length=len(request.message)
        )
        db.add(user_message)
        
        # Get AI response using same logic as new endpoint
        model = genai.GenerativeModel('gemini-2.5-flash')
        system_prompt = """You are a helpful Korean language tutor named 선생님 (Teacher). Help users learn Korean vocabulary, grammar, pronunciation, and culture."""
        
        response = model.generate_content(f"{system_prompt}\n\nUser: {request.message}\n\nPlease respond as the Korean tutor:")
        ai_response_content = response.text
        
        ai_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=ai_response_content,
            created_at=datetime.utcnow(),
            content_length=len(ai_response_content)
        )
        db.add(ai_message)
        
        conversation.message_count += 2
        conversation.updated_at = datetime.utcnow()
        db.commit()
        
        return ChatResponse(
            id=ai_message.id,
            message=request.message,
            response=ai_response_content,
            timestamp=ai_message.created_at.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Legacy chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat message: {str(e)}")

@app.get("/api/chat/history", response_model=ChatHistoryResponse)
def get_chat_history(limit: int = 50, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Legacy chat history endpoint - converts new format to old format"""
    # Get recent messages from all conversations
    messages = db.query(Message)\
        .join(Conversation)\
        .filter(
            Conversation.user_id == current_user.id,
            Conversation.is_active == True
        )\
        .order_by(Message.created_at.desc())\
        .limit(limit * 2).all()  # Get more since we'll group by pairs
    
    # Convert to legacy format (group user+assistant pairs)
    legacy_messages = []
    i = 0
    while i < len(messages) - 1:
        if messages[i].role == "assistant" and messages[i+1].role == "user":
            # Swap order to user first, then assistant
            user_msg = messages[i+1]
            ai_msg = messages[i]
            legacy_messages.append(ChatMessageResponse(
                id=ai_msg.id,
                user_id=user_msg.conversation.user_id,
                message=user_msg.content,
                response=ai_msg.content,
                timestamp=ai_msg.created_at.isoformat(),
                created_at=ai_msg.created_at.isoformat()
            ))
            i += 2
        else:
            i += 1
    
    return ChatHistoryResponse(
        messages=list(reversed(legacy_messages)),  # Reverse to show oldest first
        total=len(legacy_messages),
        limit=limit
    )

@app.post("/api/payment/submit", response_model=PaymentResponse)
def submit_payment(payment: PaymentSubmit, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        existing_payment = db.query(PaymentVerification).filter(
            PaymentVerification.transaction_id == payment.transaction_id
        ).first()
        
        if existing_payment:
            raise HTTPException(status_code=400, detail="Transaction ID already exists")
        
        new_payment = PaymentVerification(
            user_id=current_user.id,
            transaction_id=payment.transaction_id,
            amount=payment.amount,
            status="pending"
        )
        db.add(new_payment)
        db.commit()
        db.refresh(new_payment)
        
        logger.info(f"Payment submitted for user {current_user.email}: {payment.transaction_id}")
        return new_payment
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Payment submission error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to submit payment")

@app.get("/api/payment/status", response_model=List[PaymentResponse])
def get_payment_status(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    payments = db.query(PaymentVerification).filter(
        PaymentVerification.user_id == current_user.id
    ).order_by(PaymentVerification.created_at.desc()).all()
    return payments

@app.post("/api/admin/verify-payment/{payment_id}")
def verify_payment(payment_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        if not current_user.is_premium:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        payment = db.query(PaymentVerification).filter(PaymentVerification.id == payment_id).first()
        if not payment:
            raise HTTPException(status_code=404, detail="Payment not found")
        
        payment.status = "verified"
        payment.verified_at = datetime.utcnow()
        
        user = db.query(User).filter(User.id == payment.user_id).first()
        if user:
            user.is_premium = True
        
        db.commit()
        
        logger.info(f"Payment verified by admin {current_user.email}: payment_id {payment_id}")
        return {"message": "Payment verified successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Payment verification error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to verify payment")

# Admin Vocabulary Management Endpoints
@app.post("/api/admin/vocabulary", response_model=VocabularyResponse)
def create_vocabulary(vocab: VocabularyCreate, db: Session = Depends(get_db), admin_user: User = Depends(get_admin_user)):
    import json
    try:
        
        new_vocab = Vocabulary(
            korean=vocab.korean,
            english=vocab.english,
            pronunciation=vocab.pronunciation,
            examples=json.dumps(vocab.examples),
            category=vocab.category,
            difficulty_level=vocab.difficulty_level,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(new_vocab)
        db.commit()
        db.refresh(new_vocab)
        
        logger.info(f"Vocabulary created by {admin_user.email}: {vocab.korean}")
        
        return VocabularyResponse(
            id=new_vocab.id,
            korean=new_vocab.korean,
            english=new_vocab.english,
            pronunciation=new_vocab.pronunciation,
            examples=vocab.examples,
            category=new_vocab.category,
            difficulty_level=new_vocab.difficulty_level,
            created_at=new_vocab.created_at,
            updated_at=new_vocab.updated_at
        )
    except Exception as e:
        logger.error(f"Vocabulary creation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create vocabulary")

@app.put("/api/admin/vocabulary/{vocabulary_id}", response_model=VocabularyResponse)
def update_vocabulary(vocabulary_id: int, vocab: VocabularyUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    import json
    try:
        existing_vocab = db.query(Vocabulary).filter(Vocabulary.id == vocabulary_id).first()
        if not existing_vocab:
            raise HTTPException(status_code=404, detail="Vocabulary not found")
        
        if vocab.korean is not None:
            existing_vocab.korean = vocab.korean
        if vocab.english is not None:
            existing_vocab.english = vocab.english
        if vocab.pronunciation is not None:
            existing_vocab.pronunciation = vocab.pronunciation
        if vocab.examples is not None:
            existing_vocab.examples = json.dumps(vocab.examples)
        if vocab.category is not None:
            existing_vocab.category = vocab.category
        if vocab.difficulty_level is not None:
            existing_vocab.difficulty_level = vocab.difficulty_level
        
        existing_vocab.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(existing_vocab)
        
        logger.info(f"Vocabulary updated by {current_user.email}: {vocabulary_id}")
        
        return VocabularyResponse(
            id=existing_vocab.id,
            korean=existing_vocab.korean,
            english=existing_vocab.english,
            pronunciation=existing_vocab.pronunciation,
            examples=json.loads(existing_vocab.examples) if existing_vocab.examples else [],
            category=existing_vocab.category,
            difficulty_level=existing_vocab.difficulty_level,
            created_at=existing_vocab.created_at,
            updated_at=existing_vocab.updated_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vocabulary update error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update vocabulary")

@app.delete("/api/admin/vocabulary/{vocabulary_id}")
def delete_vocabulary(vocabulary_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        vocab = db.query(Vocabulary).filter(Vocabulary.id == vocabulary_id).first()
        if not vocab:
            raise HTTPException(status_code=404, detail="Vocabulary not found")
        
        db.delete(vocab)
        db.commit()
        
        logger.info(f"Vocabulary deleted by {current_user.email}: {vocabulary_id}")
        return {"message": "Vocabulary deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vocabulary deletion error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete vocabulary")

# Admin Grammar Management Endpoints
@app.post("/api/admin/grammar", response_model=GrammarResponse)
def create_grammar(grammar: GrammarCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    import json
    try:
        new_grammar = Grammar(
            title=grammar.title,
            explanation=grammar.explanation,
            examples=json.dumps(grammar.examples),
            category=grammar.category,
            difficulty_level=grammar.difficulty_level,
            created_at=datetime.utcnow()
        )
        db.add(new_grammar)
        db.commit()
        db.refresh(new_grammar)
        
        logger.info(f"Grammar created by {current_user.email}: {grammar.title}")
        
        return GrammarResponse(
            id=new_grammar.id,
            title=new_grammar.title,
            category=new_grammar.category,
            explanation=new_grammar.explanation,
            examples=grammar.examples,
            difficulty_level=new_grammar.difficulty_level,
            created_at=new_grammar.created_at,
            updated_at=new_grammar.created_at
        )
    except Exception as e:
        logger.error(f"Grammar creation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create grammar")

@app.put("/api/admin/grammar/{grammar_id}", response_model=GrammarResponse)
def update_grammar(grammar_id: int, grammar: GrammarUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    import json
    try:
        existing_grammar = db.query(Grammar).filter(Grammar.id == grammar_id).first()
        if not existing_grammar:
            raise HTTPException(status_code=404, detail="Grammar not found")
        
        if grammar.title is not None:
            existing_grammar.title = grammar.title
        if grammar.explanation is not None:
            existing_grammar.explanation = grammar.explanation
        if grammar.examples is not None:
            existing_grammar.examples = json.dumps(grammar.examples)
        if grammar.category is not None:
            existing_grammar.category = grammar.category
        if grammar.difficulty_level is not None:
            existing_grammar.difficulty_level = grammar.difficulty_level
        
        # Note: updated_at column doesn't exist in the grammar table
        db.commit()
        db.refresh(existing_grammar)
        
        logger.info(f"Grammar updated by {current_user.email}: {grammar_id}")
        
        return GrammarResponse(
            id=existing_grammar.id,
            title=existing_grammar.title,
            category=existing_grammar.category,
            explanation=existing_grammar.explanation,
            examples=json.loads(existing_grammar.examples) if isinstance(existing_grammar.examples, str) else existing_grammar.examples,
            difficulty_level=existing_grammar.difficulty_level,
            created_at=existing_grammar.created_at,
            updated_at=existing_grammar.created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Grammar update error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update grammar")

@app.delete("/api/admin/grammar/{grammar_id}")
def delete_grammar(grammar_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        grammar = db.query(Grammar).filter(Grammar.id == grammar_id).first()
        if not grammar:
            raise HTTPException(status_code=404, detail="Grammar not found")
        
        db.delete(grammar)
        db.commit()
        
        logger.info(f"Grammar deleted by {current_user.email}: {grammar_id}")
        return {"message": "Grammar deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Grammar deletion error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete grammar")

# Admin Category Management Endpoints
@app.get("/api/admin/categories", response_model=List[CategoryResponse])
def get_categories(type: Optional[str] = None, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        query = db.query(Category).order_by(Category.order)
        if type:
            query = query.filter(Category.type == type)
        categories = query.all()
        return categories
    except Exception as e:
        logger.error(f"Categories fetch error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch categories")

@app.post("/api/admin/categories", response_model=CategoryResponse)
def create_category(category: CategoryCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        new_category = Category(
            name=category.name,
            description=category.description,
            type=category.type,
            color=category.color,
            order=category.order,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(new_category)
        db.commit()
        db.refresh(new_category)
        
        logger.info(f"Category created by {current_user.email}: {category.name}")
        return new_category
    except Exception as e:
        logger.error(f"Category creation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create category")

@app.put("/api/admin/categories/{category_id}", response_model=CategoryResponse)
def update_category(category_id: int, category: CategoryUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        existing_category = db.query(Category).filter(Category.id == category_id).first()
        if not existing_category:
            raise HTTPException(status_code=404, detail="Category not found")
        
        if category.name is not None:
            existing_category.name = category.name
        if category.description is not None:
            existing_category.description = category.description
        if category.type is not None:
            existing_category.type = category.type
        if category.color is not None:
            existing_category.color = category.color
        if category.order is not None:
            existing_category.order = category.order
        
        existing_category.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(existing_category)
        
        logger.info(f"Category updated by {current_user.email}: {category_id}")
        return existing_category
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Category update error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update category")

@app.get("/api/admin/categories/{category_id}/usage")
def get_category_usage(category_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Get detailed usage information for a category"""
    try:
        category = db.query(Category).filter(Category.id == category_id).first()
        if not category:
            raise HTTPException(status_code=404, detail="Category not found")
        
        # Count vocabulary items using this category
        vocab_count = db.query(Vocabulary).filter(Vocabulary.category == category.name.lower()).count()
        
        # Count grammar items using this category
        grammar_count = db.query(Grammar).filter(Grammar.category == category.name.lower()).count()
        
        return {
            "category_id": category_id,
            "category_name": category.name,
            "vocabulary_count": vocab_count,
            "grammar_count": grammar_count,
            "total_count": vocab_count + grammar_count,
            "has_content": (vocab_count + grammar_count) > 0
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Category usage check error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to check category usage")

@app.post("/api/admin/categories/{category_id}/reassign")
def reassign_category_content(
    category_id: int, 
    target_category_id: int, 
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    """Reassign all content from one category to another"""
    try:
        # Validate source category exists
        source_category = db.query(Category).filter(Category.id == category_id).first()
        if not source_category:
            raise HTTPException(status_code=404, detail="Source category not found")
        
        # Validate target category exists
        target_category = db.query(Category).filter(Category.id == target_category_id).first()
        if not target_category:
            raise HTTPException(status_code=404, detail="Target category not found")
        
        # Ensure categories are the same type
        if source_category.type != target_category.type:
            raise HTTPException(status_code=400, detail="Cannot reassign between different category types")
        
        # Update vocabulary items
        vocab_updated = 0
        if source_category.type == 'vocabulary':
            vocab_items = db.query(Vocabulary).filter(Vocabulary.category == source_category.name.lower()).all()
            for item in vocab_items:
                item.category = target_category.name.lower()
                item.updated_at = datetime.utcnow()
            vocab_updated = len(vocab_items)
        
        # Update grammar items
        grammar_updated = 0
        if source_category.type == 'grammar':
            grammar_items = db.query(Grammar).filter(Grammar.category == source_category.name.lower()).all()
            for item in grammar_items:
                item.category = target_category.name.lower()
                item.updated_at = datetime.utcnow()
            grammar_updated = len(grammar_items)
        
        db.commit()
        
        logger.info(f"Category content reassigned by {current_user.email}: {source_category.name} -> {target_category.name}")
        
        return {
            "message": "Content reassigned successfully",
            "vocabulary_updated": vocab_updated,
            "grammar_updated": grammar_updated,
            "total_updated": vocab_updated + grammar_updated,
            "source_category": source_category.name,
            "target_category": target_category.name
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Category reassignment error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to reassign category content")

@app.delete("/api/admin/categories/{category_id}")
def delete_category(
    category_id: int, 
    force: bool = False,
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    """Delete a category with optional force flag"""
    try:
        category = db.query(Category).filter(Category.id == category_id).first()
        if not category:
            raise HTTPException(status_code=404, detail="Category not found")
        
        # Check for associated content unless force flag is used
        if not force:
            vocab_count = db.query(Vocabulary).filter(Vocabulary.category == category.name.lower()).count()
            grammar_count = db.query(Grammar).filter(Grammar.category == category.name.lower()).count()
            total_content = vocab_count + grammar_count
            
            if total_content > 0:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Cannot delete category with {total_content} associated items. Use reassignment or force deletion."
                )
        
        db.delete(category)
        db.commit()
        
        logger.info(f"Category deleted by {current_user.email}: {category_id} (force={force})")
        return {"message": "Category deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Category deletion error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete category")

@app.get("/api/admin/categories/orphaned")
def find_orphaned_content(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Find vocabulary and grammar content with orphaned categories"""
    try:
        # Get all category names that exist in the categories table
        existing_categories = db.query(Category.name).all()
        existing_category_names = {cat.name.lower() for cat in existing_categories}
        
        # Find vocabulary with orphaned categories
        all_vocab = db.query(Vocabulary).all()
        orphaned_vocab = []
        vocab_categories = set()
        
        for vocab in all_vocab:
            vocab_categories.add(vocab.category)
            if vocab.category not in existing_category_names:
                orphaned_vocab.append({
                    'id': vocab.id,
                    'korean': vocab.korean,
                    'english': vocab.english,
                    'category': vocab.category
                })
        
        # Find grammar with orphaned categories
        all_grammar = db.query(Grammar).all()
        orphaned_grammar = []
        grammar_categories = set()
        
        for grammar in all_grammar:
            grammar_categories.add(grammar.category)
            if grammar.category not in existing_category_names:
                orphaned_grammar.append({
                    'id': grammar.id,
                    'title': grammar.title,
                    'category': grammar.category
                })
        
        # Find unique orphaned categories
        orphaned_categories = (vocab_categories | grammar_categories) - existing_category_names
        
        return {
            'orphaned_vocabulary': orphaned_vocab,
            'orphaned_grammar': orphaned_grammar,
            'orphaned_categories': list(orphaned_categories),
            'summary': {
                'orphaned_vocab_count': len(orphaned_vocab),
                'orphaned_grammar_count': len(orphaned_grammar),
                'orphaned_category_count': len(orphaned_categories),
                'total_orphaned_items': len(orphaned_vocab) + len(orphaned_grammar)
            }
        }
    except Exception as e:
        logger.error(f"Orphaned content check error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to check for orphaned content")

@app.post("/api/admin/categories/cleanup-orphaned")
def cleanup_orphaned_content(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Create categories for orphaned content"""
    try:
        # Get orphaned content first
        orphaned_data = find_orphaned_content(db, current_user)
        orphaned_categories = orphaned_data['orphaned_categories']
        
        if not orphaned_categories:
            return {"message": "No orphaned categories found", "created_count": 0}
        
        # Color mapping
        colors = [
            '#3B82F6', '#10B981', '#F59E0B', '#EF4444', 
            '#8B5CF6', '#06B6D4', '#F97316', '#84CC16',
            '#EC4899', '#6B7280'
        ]
        
        created_categories = []
        
        for i, category_name in enumerate(orphaned_categories):
            # Determine type based on content
            has_vocab = db.query(Vocabulary).filter(Vocabulary.category == category_name).first() is not None
            has_grammar = db.query(Grammar).filter(Grammar.category == category_name).first() is not None
            
            # Default to vocabulary if it has vocabulary items, otherwise grammar
            category_type = 'vocabulary' if has_vocab else 'grammar'
            
            # Get next order for this type
            max_order = db.query(func.max(Category.order)).filter(Category.type == category_type).scalar() or 0
            
            # Create new category
            new_category = Category(
                name=category_name.title(),
                description=f'Auto-created category for {category_name} content',
                type=category_type,
                color=colors[i % len(colors)],
                order=max_order + 1,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            db.add(new_category)
            created_categories.append({
                'name': new_category.name,
                'type': new_category.type,
                'color': new_category.color
            })
        
        db.commit()
        
        logger.info(f"Orphaned content cleanup by {current_user.email}: {len(created_categories)} categories created")
        
        return {
            "message": "Orphaned content cleanup completed",
            "created_count": len(created_categories),
            "created_categories": created_categories
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Orphaned content cleanup error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to cleanup orphaned content")

if __name__ == "__main__":
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)