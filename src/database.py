import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict
from .evaluator import QuestionAttempt, SessionStats
import pandas as pd
import os

class DatabaseManager:
    """Manages SQLite database for storing user progress and session data"""
    
    def __init__(self, db_path: str = "ml_trainer.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                total_questions INTEGER DEFAULT 0,
                correct_answers INTEGER DEFAULT 0,
                total_score REAL DEFAULT 0.0,
                max_possible_score REAL DEFAULT 0.0,
                accuracy REAL DEFAULT 0.0,
                average_time REAL DEFAULT 0.0,
                categories_covered TEXT,  -- JSON array
                difficulty_breakdown TEXT,  -- JSON object
                session_notes TEXT
            )
        """)
        
        # Question attempts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS question_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                question_id TEXT NOT NULL,
                category TEXT NOT NULL,
                question_type TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                user_answer TEXT,
                correct BOOLEAN NOT NULL,
                score REAL NOT NULL,
                max_score REAL NOT NULL,
                time_taken REAL NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                feedback TEXT,
                ai_evaluated BOOLEAN DEFAULT FALSE,
                question_data TEXT,  -- JSON of full question data
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        """)
        
        # User achievements table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS achievements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                achievement_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                earned_date TIMESTAMP NOT NULL,
                session_id INTEGER,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        """)
        
        # User preferences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Category performance tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS category_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                total_attempts INTEGER DEFAULT 0,
                correct_attempts INTEGER DEFAULT 0,
                total_score REAL DEFAULT 0.0,
                max_possible_score REAL DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(category, difficulty)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_session(self, start_time: datetime = None) -> int:
        """Create a new session and return its ID"""
        if start_time is None:
            start_time = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sessions (start_time) VALUES (?)
        """, (start_time,))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return session_id
    
    def update_session(self, session_id: int, session_stats: SessionStats):
        """Update session with final statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE sessions SET
                end_time = ?,
                total_questions = ?,
                correct_answers = ?,
                total_score = ?,
                max_possible_score = ?,
                accuracy = ?,
                average_time = ?,
                categories_covered = ?,
                difficulty_breakdown = ?
            WHERE id = ?
        """, (
            session_stats.end_time,
            session_stats.total_questions,
            session_stats.correct_answers,
            session_stats.total_score,
            session_stats.max_possible_score,
            session_stats.accuracy,
            session_stats.average_time,
            json.dumps(session_stats.categories_covered),
            json.dumps(session_stats.difficulty_breakdown),
            session_id
        ))
        
        conn.commit()
        conn.close()
    
    def record_attempt(self, session_id: int, attempt: QuestionAttempt, question_data: Dict = None):
        """Record a question attempt"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO question_attempts (
                session_id, question_id, category, question_type, difficulty,
                user_answer, correct, score, max_score, time_taken,
                timestamp, feedback, ai_evaluated, question_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            attempt.question_id,
            attempt.category,
            attempt.question_type,
            attempt.difficulty,
            attempt.user_answer,
            attempt.correct,
            attempt.score,
            attempt.max_score,
            attempt.time_taken,
            attempt.timestamp,
            attempt.feedback,
            attempt.ai_evaluated,
            json.dumps(question_data) if question_data else None
        ))
        
        # Update category performance
        self._update_category_performance(cursor, attempt)
        
        conn.commit()
        conn.close()
    
    def _update_category_performance(self, cursor, attempt: QuestionAttempt):
        """Update category performance statistics"""
        
        # Check if record exists
        cursor.execute("""
            SELECT total_attempts, correct_attempts, total_score, max_possible_score
            FROM category_performance 
            WHERE category = ? AND difficulty = ?
        """, (attempt.category, attempt.difficulty))
        
        result = cursor.fetchone()
        
        if result:
            # Update existing record
            total_attempts, correct_attempts, total_score, max_possible_score = result
            
            cursor.execute("""
                UPDATE category_performance SET
                    total_attempts = ?,
                    correct_attempts = ?,
                    total_score = ?,
                    max_possible_score = ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE category = ? AND difficulty = ?
            """, (
                total_attempts + 1,
                correct_attempts + (1 if attempt.correct else 0),
                total_score + attempt.score,
                max_possible_score + attempt.max_score,
                attempt.category,
                attempt.difficulty
            ))
        else:
            # Create new record
            cursor.execute("""
                INSERT INTO category_performance (
                    category, difficulty, total_attempts, correct_attempts,
                    total_score, max_possible_score
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                attempt.category,
                attempt.difficulty,
                1,
                1 if attempt.correct else 0,
                attempt.score,
                attempt.max_score
            ))
    
    def record_achievement(self, achievement: Dict, session_id: int):
        """Record an earned achievement"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO achievements (achievement_type, title, description, earned_date, session_id)
            VALUES (?, ?, ?, ?, ?)
        """, (
            achievement["type"],
            achievement["title"],
            achievement["description"],
            datetime.now(),
            session_id
        ))
        
        conn.commit()
        conn.close()
    
    def get_session_history(self, limit: int = 10) -> List[Dict]:
        """Get recent session history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM sessions 
            WHERE end_time IS NOT NULL
            ORDER BY start_time DESC 
            LIMIT ?
        """, (limit,))
        
        columns = [description[0] for description in cursor.description]
        sessions = []
        
        for row in cursor.fetchall():
            session = dict(zip(columns, row))
            # Parse JSON fields
            if session['categories_covered']:
                session['categories_covered'] = json.loads(session['categories_covered'])
            if session['difficulty_breakdown']:
                session['difficulty_breakdown'] = json.loads(session['difficulty_breakdown'])
            sessions.append(session)
        
        conn.close()
        return sessions
    
    def get_category_performance(self) -> pd.DataFrame:
        """Get performance statistics by category and difficulty"""
        conn = sqlite3.connect(self.db_path)
        
        df = pd.read_sql_query("""
            SELECT 
                category,
                difficulty,
                total_attempts,
                correct_attempts,
                CAST(correct_attempts AS REAL) / total_attempts * 100 as accuracy,
                total_score / max_possible_score * 100 as avg_score_percentage,
                last_updated
            FROM category_performance
            WHERE total_attempts > 0
            ORDER BY category, difficulty
        """, conn)
        
        conn.close()
        return df
    
    def get_progress_over_time(self, days: int = 30) -> pd.DataFrame:
        """Get progress data over specified number of days"""
        conn = sqlite3.connect(self.db_path)
        
        since_date = datetime.now() - timedelta(days=days)
        
        df = pd.read_sql_query("""
            SELECT 
                DATE(start_time) as date,
                COUNT(*) as sessions,
                SUM(total_questions) as total_questions,
                SUM(correct_answers) as correct_answers,
                AVG(accuracy) as avg_accuracy,
                AVG(average_time) as avg_time_per_question
            FROM sessions 
            WHERE start_time >= ? AND end_time IS NOT NULL
            GROUP BY DATE(start_time)
            ORDER BY date
        """, conn, params=(since_date,))
        
        conn.close()
        return df
    
    def get_weak_areas_from_history(self, min_attempts: int = 3) -> List[Dict]:
        """Get weak areas based on historical performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                category,
                COUNT(*) as attempts,
                SUM(CASE WHEN correct THEN 1 ELSE 0 END) as correct,
                AVG(score / max_score * 100) as avg_score
            FROM question_attempts
            GROUP BY category
            HAVING attempts >= ?
            ORDER BY AVG(score / max_score * 100) ASC
        """, (min_attempts,))
        
        weak_areas = []
        for row in cursor.fetchall():
            category, attempts, correct, avg_score = row
            accuracy = (correct / attempts) * 100
            
            if accuracy < 75:  # Consider weak if < 75% accuracy
                weak_areas.append({
                    "category": category,
                    "attempts": attempts,
                    "accuracy": accuracy,
                    "avg_score": avg_score
                })
        
        conn.close()
        return weak_areas
    
    def get_achievements(self, limit: int = None) -> List[Dict]:
        """Get user achievements"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM achievements ORDER BY earned_date DESC"
        params = ()
        
        if limit:
            query += " LIMIT ?"
            params = (limit,)
        
        cursor.execute(query, params)
        columns = [description[0] for description in cursor.description]
        achievements = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return achievements
    
    def set_preference(self, key: str, value: str):
        """Set a user preference"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO user_preferences (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (key, value))
        
        conn.commit()
        conn.close()
    
    def get_preference(self, key: str, default: str = None) -> Optional[str]:
        """Get a user preference"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT value FROM user_preferences WHERE key = ?", (key,))
        result = cursor.fetchone()
        
        conn.close()
        return result[0] if result else default
    
    def get_total_stats(self) -> Dict:
        """Get overall statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total questions answered
        cursor.execute("SELECT COUNT(*) FROM question_attempts")
        total_questions = cursor.fetchone()[0]
        
        # Total correct answers
        cursor.execute("SELECT COUNT(*) FROM question_attempts WHERE correct = 1")
        total_correct = cursor.fetchone()[0]
        
        # Total sessions
        cursor.execute("SELECT COUNT(*) FROM sessions WHERE end_time IS NOT NULL")
        total_sessions = cursor.fetchone()[0]
        
        # Total study time (sum of average_time * total_questions for each session)
        cursor.execute("""
            SELECT SUM(average_time * total_questions) 
            FROM sessions 
            WHERE end_time IS NOT NULL AND average_time IS NOT NULL
        """)
        result = cursor.fetchone()[0]
        total_study_time = result if result else 0
        
        # Categories practiced
        cursor.execute("SELECT COUNT(DISTINCT category) FROM question_attempts")
        categories_practiced = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_questions": total_questions,
            "total_correct": total_correct,
            "overall_accuracy": (total_correct / total_questions * 100) if total_questions > 0 else 0,
            "total_sessions": total_sessions,
            "total_study_time_minutes": total_study_time / 60,
            "categories_practiced": categories_practiced
        }
    
    def export_data(self, filepath: str):
        """Export all data to JSON file"""
        conn = sqlite3.connect(self.db_path)
        
        # Get all data
        sessions = pd.read_sql_query("SELECT * FROM sessions", conn)
        attempts = pd.read_sql_query("SELECT * FROM question_attempts", conn)
        achievements = pd.read_sql_query("SELECT * FROM achievements", conn)
        preferences = pd.read_sql_query("SELECT * FROM user_preferences", conn)
        
        export_data = {
            "export_date": datetime.now().isoformat(),
            "sessions": sessions.to_dict('records'),
            "question_attempts": attempts.to_dict('records'),
            "achievements": achievements.to_dict('records'),
            "preferences": preferences.to_dict('records')
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        conn.close()
    
    def backup_database(self, backup_path: str):
        """Create a backup of the database"""
        import shutil
        shutil.copy2(self.db_path, backup_path)