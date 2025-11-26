"""
Free Spaced Repetition Scheduler (FSRS) Algorithm Implementation
Based on the research-backed FSRS algorithm for optimal spaced repetition
"""

import math
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple

class CardState(Enum):
    NEW = "new"
    LEARNING = "learning"
    REVIEW = "review"
    RELEARNING = "relearning"

class Grade(Enum):
    AGAIN = 1    # Failed recall
    HARD = 2     # Difficult recall
    GOOD = 3     # Normal recall
    EASY = 4     # Easy recall

@dataclass
class FSRSCard:
    """Represents an FSRS spaced repetition card"""
    id: Optional[int] = None
    user_id: int = 0
    content_type: str = "vocabulary"  # vocabulary or grammar
    content_id: int = 0
    
    # FSRS Core Parameters
    stability: float = 2.0      # Memory strength in days
    difficulty: float = 5.0     # Intrinsic difficulty (0-10)
    retrievability: float = 0.9 # Current recall probability (0-1)
    
    # Card State
    state: CardState = CardState.NEW
    due_date: datetime = None
    last_review: Optional[datetime] = None
    
    # Scheduling Info
    interval_days: int = 1
    review_count: int = 0
    lapse_count: int = 0
    
    def __post_init__(self):
        if self.due_date is None:
            self.due_date = datetime.now()

@dataclass
class ReviewResult:
    """Result of a review session"""
    grade: Grade
    response_time_ms: Optional[int] = None
    review_date: datetime = None
    
    def __post_init__(self):
        if self.review_date is None:
            self.review_date = datetime.now()

@dataclass
class FSRSParameters:
    """FSRS algorithm parameters - can be customized per user"""
    # Learning parameters
    learning_steps: List[int] = None  # Minutes for learning steps
    graduating_interval: int = 1      # Days to graduate from learning
    easy_interval: int = 4           # Days for easy button in learning
    
    # Algorithm weights (research-optimized defaults)
    w: List[float] = None
    
    # Stability calculation parameters
    max_interval: int = 36500        # Maximum interval (100 years)
    min_interval: int = 1            # Minimum interval
    
    def __post_init__(self):
        if self.learning_steps is None:
            self.learning_steps = [1, 10]  # 1 minute, 10 minutes
        
        if self.w is None:
            # Research-optimized weights for FSRS v4
            self.w = [
                0.4072, 1.1829, 3.1262, 15.4722, 7.2102,
                0.5316, 1.0651, 0.0234, 1.616, 0.1544,
                1.0824, 1.9813, 0.0953, 0.2975, 2.2042,
                0.2407, 2.9466, 0.5034, 0.6567
            ]

class FSRSAlgorithm:
    """Core FSRS algorithm implementation"""
    
    def __init__(self, parameters: Optional[FSRSParameters] = None):
        self.params = parameters or FSRSParameters()
    
    def schedule_card(self, card: FSRSCard, review: ReviewResult) -> FSRSCard:
        """
        Schedule a card based on review result
        Returns updated card with new parameters
        """
        new_card = FSRSCard(**card.__dict__)  # Create copy
        
        # Update review metadata
        new_card.last_review = review.review_date
        new_card.review_count += 1
        
        if card.state == CardState.NEW:
            return self._schedule_new_card(new_card, review)
        elif card.state == CardState.LEARNING:
            return self._schedule_learning_card(new_card, review)
        elif card.state == CardState.REVIEW:
            return self._schedule_review_card(new_card, review)
        elif card.state == CardState.RELEARNING:
            return self._schedule_relearning_card(new_card, review)
        
        return new_card
    
    def _schedule_new_card(self, card: FSRSCard, review: ReviewResult) -> FSRSCard:
        """Schedule a new card based on first review"""
        card.difficulty = self._init_difficulty(review.grade)
        
        if review.grade == Grade.AGAIN:
            card.state = CardState.LEARNING
            card.interval_days = 0
            card.due_date = review.review_date + timedelta(minutes=self.params.learning_steps[0])
            card.lapse_count += 1
        elif review.grade == Grade.HARD:
            card.state = CardState.LEARNING
            card.stability = self._init_stability(review.grade)
            card.interval_days = 0
            card.due_date = review.review_date + timedelta(minutes=self.params.learning_steps[0])
        elif review.grade == Grade.GOOD:
            card.state = CardState.REVIEW
            card.stability = self._init_stability(review.grade)
            card.interval_days = self.params.graduating_interval
            card.due_date = review.review_date + timedelta(days=card.interval_days)
        elif review.grade == Grade.EASY:
            card.state = CardState.REVIEW
            card.stability = self._init_stability(review.grade)
            card.interval_days = self.params.easy_interval
            card.due_date = review.review_date + timedelta(days=card.interval_days)
        
        return card
    
    def _schedule_learning_card(self, card: FSRSCard, review: ReviewResult) -> FSRSCard:
        """Schedule a learning card"""
        if review.grade == Grade.AGAIN:
            # Reset to first learning step
            card.due_date = review.review_date + timedelta(minutes=self.params.learning_steps[0])
            card.lapse_count += 1
        elif review.grade in [Grade.HARD, Grade.GOOD]:
            # Move to next learning step or graduate
            if len(self.params.learning_steps) > 1:
                card.due_date = review.review_date + timedelta(minutes=self.params.learning_steps[1])
            else:
                # Graduate to review
                card.state = CardState.REVIEW
                card.stability = self._calculate_stability(card, review.grade)
                card.interval_days = self.params.graduating_interval
                card.due_date = review.review_date + timedelta(days=card.interval_days)
        elif review.grade == Grade.EASY:
            # Graduate immediately with easy interval
            card.state = CardState.REVIEW
            card.stability = self._calculate_stability(card, review.grade)
            card.interval_days = self.params.easy_interval
            card.due_date = review.review_date + timedelta(days=card.interval_days)
        
        return card
    
    def _schedule_review_card(self, card: FSRSCard, review: ReviewResult) -> FSRSCard:
        """Schedule a review card"""
        if review.grade == Grade.AGAIN:
            # Move to relearning
            card.state = CardState.RELEARNING
            card.due_date = review.review_date + timedelta(minutes=self.params.learning_steps[0])
            card.lapse_count += 1
        else:
            # Update difficulty and stability
            card.difficulty = self._update_difficulty(card, review.grade)
            card.stability = self._calculate_stability(card, review.grade)
            card.interval_days = self._calculate_interval(card.stability)
            card.due_date = review.review_date + timedelta(days=card.interval_days)
        
        return card
    
    def _schedule_relearning_card(self, card: FSRSCard, review: ReviewResult) -> FSRSCard:
        """Schedule a relearning card"""
        if review.grade == Grade.AGAIN:
            # Stay in relearning, reset to first step
            card.due_date = review.review_date + timedelta(minutes=self.params.learning_steps[0])
        elif review.grade in [Grade.HARD, Grade.GOOD, Grade.EASY]:
            # Graduate back to review
            card.state = CardState.REVIEW
            card.stability = self._calculate_stability(card, review.grade)
            card.interval_days = max(1, int(card.stability * 0.25))  # Conservative interval after relearning
            card.due_date = review.review_date + timedelta(days=card.interval_days)
        
        return card
    
    def _init_difficulty(self, grade: Grade) -> float:
        """Calculate initial difficulty for a new card"""
        w = self.params.w
        if grade == Grade.EASY:
            return w[4] - (w[5] * 3)
        elif grade == Grade.GOOD:
            return w[4] - (w[5] * 2)
        elif grade == Grade.HARD:
            return w[4] - w[5]
        else:  # Grade.AGAIN
            return w[4]
    
    def _init_stability(self, grade: Grade) -> float:
        """Calculate initial stability for a new card"""
        w = self.params.w
        return max(w[grade.value - 1], 0.1)
    
    def _update_difficulty(self, card: FSRSCard, grade: Grade) -> float:
        """Update difficulty based on review performance"""
        w = self.params.w
        delta_d = w[6] * (grade.value - 3)
        difficulty = card.difficulty - delta_d * w[7]
        return max(1, min(10, difficulty))
    
    def _calculate_stability(self, card: FSRSCard, grade: Grade) -> float:
        """Calculate new stability using FSRS formula"""
        w = self.params.w
        
        if card.state == CardState.NEW:
            return self._init_stability(grade)
        
        # Calculate stability increment based on retrievability and grade
        retrievability = self._calculate_retrievability(card)
        
        if grade == Grade.AGAIN:
            new_s = w[11] * (card.difficulty ** (-w[12])) * ((card.stability + 1) ** w[13] - 1) * math.exp(w[14] * (1 - retrievability))
        else:
            new_s = card.stability * (
                1 + math.exp(w[8]) *
                (11 - card.difficulty) *
                (card.stability ** (-w[9])) *
                (math.exp((1 - retrievability) * w[10]) - 1)
            )
        
        return max(0.1, min(self.params.max_interval, new_s))
    
    def _calculate_retrievability(self, card: FSRSCard) -> float:
        """Calculate current retrievability (probability of recall)"""
        if card.last_review is None:
            return 0.9
        
        days_since_review = (datetime.now() - card.last_review).days
        return (1 + days_since_review / (9 * card.stability)) ** (-1)
    
    def _calculate_interval(self, stability: float) -> int:
        """Calculate interval in days from stability"""
        interval = max(1, round(stability * 0.9))  # 90% retrievability threshold
        return min(self.params.max_interval, interval)
    
    def get_due_cards(self, cards: List[FSRSCard], limit: Optional[int] = None) -> List[FSRSCard]:
        """Get cards that are due for review, sorted by priority"""
        now = datetime.now()
        due_cards = [card for card in cards if card.due_date <= now]
        
        # Sort by priority: overdue first, then by state priority
        def priority_key(card):
            days_overdue = (now - card.due_date).days
            state_priority = {
                CardState.LEARNING: 1,
                CardState.RELEARNING: 2,
                CardState.REVIEW: 3,
                CardState.NEW: 4
            }
            return (-days_overdue, state_priority.get(card.state, 5))
        
        due_cards.sort(key=priority_key)
        
        if limit:
            return due_cards[:limit]
        
        return due_cards
    
    def calculate_retention_rate(self, reviews: List[Tuple[ReviewResult, FSRSCard]]) -> float:
        """Calculate retention rate from review history"""
        if not reviews:
            return 0.0
        
        successful_reviews = sum(1 for review, _ in reviews if review.grade != Grade.AGAIN)
        return successful_reviews / len(reviews)