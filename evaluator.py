import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass, asdict
import streamlit as st

@dataclass
class QuestionAttempt:
    """Data class for storing question attempt results"""
    question_id: str
    category: str
    question_type: str
    difficulty: str
    user_answer: str
    correct: bool
    score: float
    max_score: float
    time_taken: float  # seconds
    timestamp: datetime
    feedback: str
    ai_evaluated: bool = False

@dataclass
class SessionStats:
    """Data class for session statistics"""
    total_questions: int
    correct_answers: int
    total_score: float
    max_possible_score: float
    accuracy: float
    average_time: float
    categories_covered: List[str]
    difficulty_breakdown: Dict[str, int]
    start_time: datetime
    end_time: datetime

class PerformanceEvaluator:
    """Evaluates user performance and provides insights"""
    
    def __init__(self):
        self.current_session_attempts = []
        self.streak_count = 0
        self.best_streak = 0
        
    def record_attempt(self, attempt: QuestionAttempt) -> Dict:
        """Record a question attempt and update statistics"""
        
        # Add to current session
        self.current_session_attempts.append(attempt)
        
        # Update streak
        if attempt.correct:
            self.streak_count += 1
            self.best_streak = max(self.best_streak, self.streak_count)
        else:
            self.streak_count = 0
        
        # Calculate immediate feedback
        feedback = self._generate_immediate_feedback(attempt)
        
        return feedback
    
    def _generate_immediate_feedback(self, attempt: QuestionAttempt) -> Dict:
        """Generate immediate feedback for an attempt"""
        
        feedback = {
            "correct": attempt.correct,
            "score": attempt.score,
            "max_score": attempt.max_score,
            "percentage": (attempt.score / attempt.max_score * 100) if attempt.max_score > 0 else 0,
            "streak": self.streak_count,
            "time_taken": attempt.time_taken,
            "feedback_message": attempt.feedback
        }
        
        # Add performance level
        percentage = feedback["percentage"]
        if percentage >= 90:
            feedback["performance_level"] = "Excellent"
            feedback["performance_emoji"] = "🎉"
        elif percentage >= 75:
            feedback["performance_level"] = "Good"
            feedback["performance_emoji"] = "👍"
        elif percentage >= 60:
            feedback["performance_level"] = "Fair"
            feedback["performance_emoji"] = "👌"
        else:
            feedback["performance_level"] = "Needs Improvement"
            feedback["performance_emoji"] = "💪"
        
        return feedback
    
    def get_session_stats(self) -> SessionStats:
        """Calculate statistics for the current session"""
        
        if not self.current_session_attempts:
            return SessionStats(
                total_questions=0,
                correct_answers=0,
                total_score=0,
                max_possible_score=0,
                accuracy=0,
                average_time=0,
                categories_covered=[],
                difficulty_breakdown={},
                start_time=datetime.now(),
                end_time=datetime.now()
            )
        
        attempts = self.current_session_attempts
        
        total_questions = len(attempts)
        correct_answers = sum(1 for a in attempts if a.correct)
        total_score = sum(a.score for a in attempts)
        max_possible_score = sum(a.max_score for a in attempts)
        accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0
        average_time = sum(a.time_taken for a in attempts) / total_questions if attempts else 0
        
        categories_covered = list(set(a.category for a in attempts))
        difficulty_breakdown = {}
        for attempt in attempts:
            diff = attempt.difficulty
            difficulty_breakdown[diff] = difficulty_breakdown.get(diff, 0) + 1
        
        return SessionStats(
            total_questions=total_questions,
            correct_answers=correct_answers,
            total_score=total_score,
            max_possible_score=max_possible_score,
            accuracy=accuracy,
            average_time=average_time,
            categories_covered=categories_covered,
            difficulty_breakdown=difficulty_breakdown,
            start_time=attempts[0].timestamp if attempts else datetime.now(),
            end_time=attempts[-1].timestamp if attempts else datetime.now()
        )
    
    def get_weak_areas(self) -> List[Dict]:
        """Identify areas where user needs improvement"""
        
        if len(self.current_session_attempts) < 3:
            return []
        
        # Group by category and calculate performance
        category_performance = {}
        for attempt in self.current_session_attempts:
            cat = attempt.category
            if cat not in category_performance:
                category_performance[cat] = {"correct": 0, "total": 0, "scores": []}
            
            category_performance[cat]["total"] += 1
            if attempt.correct:
                category_performance[cat]["correct"] += 1
            category_performance[cat]["scores"].append(attempt.score / attempt.max_score * 100)
        
        # Identify weak areas (< 70% accuracy)
        weak_areas = []
        for category, stats in category_performance.items():
            accuracy = (stats["correct"] / stats["total"]) * 100
            avg_score = sum(stats["scores"]) / len(stats["scores"])
            
            if accuracy < 70 and stats["total"] >= 2:  # At least 2 attempts
                weak_areas.append({
                    "category": category,
                    "accuracy": accuracy,
                    "avg_score": avg_score,
                    "attempts": stats["total"],
                    "recommendation": self._get_recommendation(category, accuracy)
                })
        
        return sorted(weak_areas, key=lambda x: x["accuracy"])
    
    def _get_recommendation(self, category: str, accuracy: float) -> str:
        """Get specific recommendations for weak areas"""
        
        recommendations = {
            "RAG": "Review vector databases, retrieval methods, and embedding techniques",
            "Fine-tuning": "Study parameter-efficient fine-tuning methods like LoRA and prompt tuning",
            "Model Architecture": "Focus on transformer architecture, attention mechanisms, and model design patterns",
            "Data Processing": "Practice data preprocessing, tokenization, and feature engineering",
            "MLOps": "Learn about model deployment, monitoring, and CI/CD for ML systems",
            "Evaluation": "Study evaluation metrics, A/B testing, and model validation techniques",
            "Vector Databases": "Understand indexing methods, similarity search, and vector operations",
            "Transformers": "Deep dive into attention, positional encoding, and transformer variants"
        }
        
        base_rec = recommendations.get(category, f"Review fundamental concepts in {category}")
        
        if accuracy < 40:
            return f"Priority: {base_rec}. Consider starting with beginner-level resources."
        elif accuracy < 60:
            return f"Focus area: {base_rec}. Practice with intermediate-level questions."
        else:
            return f"Improvement area: {base_rec}. Challenge yourself with advanced questions."
    
    def suggest_next_difficulty(self, category: str) -> str:
        """Suggest next difficulty level based on performance"""
        
        recent_attempts = [a for a in self.current_session_attempts[-5:] 
                          if a.category == category]
        
        if not recent_attempts:
            return "beginner"
        
        accuracy = sum(1 for a in recent_attempts if a.correct) / len(recent_attempts) * 100
        
        if accuracy >= 85:
            return "advanced"
        elif accuracy >= 70:
            return "intermediate"
        else:
            return "beginner"
    
    def get_achievements(self) -> List[Dict]:
        """Check for achievements and milestones"""
        
        achievements = []
        stats = self.get_session_stats()
        
        # Session-based achievements
        if stats.total_questions >= 10:
            achievements.append({
                "title": "Question Master",
                "description": f"Answered {stats.total_questions} questions in one session",
                "emoji": "📚",
                "type": "session"
            })
        
        if stats.accuracy >= 90 and stats.total_questions >= 5:
            achievements.append({
                "title": "Excellence",
                "description": f"Achieved {stats.accuracy:.1f}% accuracy",
                "emoji": "🎯",
                "type": "performance"
            })
        
        # Streak achievements
        if self.streak_count >= 5:
            achievements.append({
                "title": "Hot Streak",
                "description": f"Correct answers streak: {self.streak_count}",
                "emoji": "🔥",
                "type": "streak"
            })
        
        if self.best_streak >= 10:
            achievements.append({
                "title": "Streak Master",
                "description": f"Best streak: {self.best_streak} correct answers",
                "emoji": "👑",
                "type": "milestone"
            })
        
        # Category coverage
        if len(stats.categories_covered) >= 5:
            achievements.append({
                "title": "Well Rounded",
                "description": f"Practiced {len(stats.categories_covered)} different topics",
                "emoji": "🌟",
                "type": "coverage"
            })
        
        return achievements
    
    def reset_session(self):
        """Reset current session data"""
        self.current_session_attempts = []
        self.streak_count = 0
    
    def export_session_data(self) -> Dict:
        """Export session data for persistence"""
        return {
            "attempts": [asdict(attempt) for attempt in self.current_session_attempts],
            "streak_count": self.streak_count,
            "best_streak": self.best_streak,
            "session_stats": asdict(self.get_session_stats())
        }

class AdaptiveDifficultyManager:
    """Manages adaptive difficulty based on user performance"""
    
    def __init__(self):
        self.category_difficulty = {}  # Track difficulty level per category
    
    def update_difficulty(self, category: str, performance_history: List[QuestionAttempt]):
        """Update difficulty level for a category based on recent performance"""
        
        if not performance_history:
            self.category_difficulty[category] = "beginner"
            return
        
        # Look at last 3 attempts in this category
        recent = [a for a in performance_history[-10:] if a.category == category][-3:]
        
        if len(recent) < 2:
            return  # Not enough data
        
        accuracy = sum(1 for a in recent if a.correct) / len(recent)
        avg_score = sum(a.score / a.max_score for a in recent) / len(recent)
        
        current_difficulty = self.category_difficulty.get(category, "beginner")
        
        # Difficulty adjustment logic
        if accuracy >= 0.8 and avg_score >= 0.8:
            # Performing well, can increase difficulty
            if current_difficulty == "beginner":
                self.category_difficulty[category] = "intermediate"
            elif current_difficulty == "intermediate":
                self.category_difficulty[category] = "advanced"
        elif accuracy < 0.5 or avg_score < 0.5:
            # Struggling, decrease difficulty
            if current_difficulty == "advanced":
                self.category_difficulty[category] = "intermediate"
            elif current_difficulty == "intermediate":
                self.category_difficulty[category] = "beginner"
    
    def get_difficulty(self, category: str) -> str:
        """Get current difficulty level for a category"""
        return self.category_difficulty.get(category, "beginner")
    
    def get_all_difficulties(self) -> Dict[str, str]:
        """Get difficulty levels for all categories"""
        return self.category_difficulty.copy()