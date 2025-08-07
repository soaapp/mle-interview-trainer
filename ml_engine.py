import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import json
import pickle
import os
from typing import List, Dict, Tuple, Optional
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import logging

class MLEngine:
    """
    Machine Learning engine for intelligent question recommendation,
    difficulty prediction, and response analysis
    """
    
    def __init__(self, model_cache_dir: str = "ml_models"):
        self.model_cache_dir = model_cache_dir
        os.makedirs(model_cache_dir, exist_ok=True)
        
        # Initialize models
        self.sentence_model = None
        self.faiss_index = None
        self.question_embeddings = None
        self.questions_df = None
        self.difficulty_predictor = None
        self.response_classifier = None
        
        # Model paths
        self.embeddings_path = os.path.join(model_cache_dir, "question_embeddings.npy")
        self.faiss_path = os.path.join(model_cache_dir, "faiss_index.index")
        self.questions_path = os.path.join(model_cache_dir, "questions_df.pkl")
        self.difficulty_model_path = os.path.join(model_cache_dir, "difficulty_model.pkl")
        self.response_model_path = os.path.join(model_cache_dir, "response_model.pkl")
        
    @st.cache_resource
    def load_sentence_transformer(_self):
        """Load sentence transformer model (cached)"""
        try:
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Error loading sentence transformer: {e}")
            return None
    
    def build_question_embeddings(self, question_bank_path: str = "question_bank.json"):
        """
        Build embeddings for all questions in the question bank
        This is the core ML component for semantic similarity
        """
        if self.sentence_model is None:
            self.sentence_model = self.load_sentence_transformer()
        
        if self.sentence_model is None:
            return False
            
        # Check if embeddings already exist
        if (os.path.exists(self.embeddings_path) and 
            os.path.exists(self.faiss_path) and 
            os.path.exists(self.questions_path)):
            self.load_embeddings()
            return True
        
        try:
            # Load question bank
            with open(question_bank_path, 'r') as f:
                question_bank = json.load(f)
            
            # Extract all questions
            questions_data = []
            
            for category, question_types in question_bank["categories"].items():
                for question_type, questions in question_types.items():
                    for question in questions:
                        questions_data.append({
                            'id': question.get('id', ''),
                            'category': category,
                            'question_type': question_type,
                            'difficulty': question.get('difficulty', 'intermediate'),
                            'question': question.get('question', ''),
                            'topics': question.get('topics', []),
                            'explanation': question.get('explanation', '')
                        })
            
            self.questions_df = pd.DataFrame(questions_data)
            
            # Create text for embedding (question + explanation + topics)
            embedding_texts = []
            for _, row in self.questions_df.iterrows():
                text = f"{row['question']} {row.get('explanation', '')} {' '.join(row.get('topics', []))}"
                embedding_texts.append(text)
            
            # Generate embeddings
            st.info("🤖 Generating question embeddings using ML model...")
            self.question_embeddings = self.sentence_model.encode(embedding_texts)
            
            # Build FAISS index for fast similarity search
            dimension = self.question_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.question_embeddings)
            self.faiss_index.add(self.question_embeddings.astype('float32'))
            
            # Save models
            self.save_embeddings()
            st.success("✅ Question embeddings built and saved!")
            return True
            
        except Exception as e:
            st.error(f"Error building embeddings: {e}")
            return False
    
    def save_embeddings(self):
        """Save embeddings and models to disk"""
        try:
            np.save(self.embeddings_path, self.question_embeddings)
            faiss.write_index(self.faiss_index, self.faiss_path)
            with open(self.questions_path, 'wb') as f:
                pickle.dump(self.questions_df, f)
        except Exception as e:
            logging.error(f"Error saving embeddings: {e}")
    
    def load_embeddings(self):
        """Load pre-built embeddings and models"""
        try:
            self.question_embeddings = np.load(self.embeddings_path)
            self.faiss_index = faiss.read_index(self.faiss_path)
            with open(self.questions_path, 'rb') as f:
                self.questions_df = pickle.load(f)
            return True
        except Exception as e:
            logging.error(f"Error loading embeddings: {e}")
            return False
    
    def find_similar_questions(self, query_question: str, k: int = 5) -> List[Dict]:
        """
        Find similar questions using semantic similarity
        Core ML feature for recommendation
        """
        if self.sentence_model is None:
            self.sentence_model = self.load_sentence_transformer()
        
        if (self.sentence_model is None or 
            self.faiss_index is None or 
            self.questions_df is None):
            return []
        
        try:
            # Embed the query question
            query_embedding = self.sentence_model.encode([query_question])
            faiss.normalize_L2(query_embedding)
            
            # Search for similar questions
            similarities, indices = self.faiss_index.search(query_embedding.astype('float32'), k + 1)
            
            similar_questions = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if i == 0:  # Skip the query question itself if it's in the database
                    continue
                
                question_data = self.questions_df.iloc[idx].to_dict()
                question_data['similarity_score'] = float(similarity)
                similar_questions.append(question_data)
            
            return similar_questions[:k]
            
        except Exception as e:
            logging.error(f"Error finding similar questions: {e}")
            return []
    
    def recommend_questions_for_weak_areas(self, user_performance_data: List[Dict], k: int = 3) -> List[Dict]:
        """
        Recommend questions based on user's weak areas using ML
        """
        if not user_performance_data:
            return []
        
        # Analyze weak categories
        category_performance = {}
        for attempt in user_performance_data:
            category = attempt.get('category', '')
            correct = attempt.get('correct', False)
            
            if category not in category_performance:
                category_performance[category] = {'correct': 0, 'total': 0}
            
            category_performance[category]['total'] += 1
            if correct:
                category_performance[category]['correct'] += 1
        
        # Find weakest categories (accuracy < 70%)
        weak_categories = []
        for category, performance in category_performance.items():
            if performance['total'] >= 2:  # At least 2 attempts
                accuracy = performance['correct'] / performance['total']
                if accuracy < 0.7:
                    weak_categories.append((category, accuracy))
        
        # Sort by accuracy (weakest first)
        weak_categories.sort(key=lambda x: x[1])
        
        recommendations = []
        if self.questions_df is not None:
            for category, _ in weak_categories[:2]:  # Top 2 weak categories
                category_questions = self.questions_df[
                    self.questions_df['category'] == category
                ].sample(min(k, len(self.questions_df[self.questions_df['category'] == category])))
                
                for _, question in category_questions.iterrows():
                    rec = question.to_dict()
                    rec['recommendation_reason'] = f"Weak area: {category}"
                    recommendations.append(rec)
        
        return recommendations[:k]
    
    def train_difficulty_predictor(self, user_attempt_data: List[Dict]) -> bool:
        """
        Train ML model to predict question difficulty based on user performance
        """
        if len(user_attempt_data) < 20:  # Need minimum data
            return False
        
        try:
            # Prepare training data
            features = []
            targets = []
            
            for attempt in user_attempt_data:
                # Features: response_time, category, question_type, user_previous_accuracy
                feature_vector = [
                    attempt.get('time_taken', 30),  # Response time
                    len(attempt.get('user_answer', '')),  # Response length
                    1 if attempt.get('question_type') == 'multiple_choice' else 0,
                    1 if attempt.get('question_type') == 'open_ended' else 0,
                    1 if attempt.get('question_type') == 'coding' else 0,
                ]
                
                # Target: difficulty level
                difficulty_map = {'beginner': 0, 'intermediate': 1, 'advanced': 2}
                difficulty = difficulty_map.get(attempt.get('difficulty', 'intermediate'), 1)
                
                features.append(feature_vector)
                targets.append(difficulty)
            
            # Train model
            X = np.array(features)
            y = np.array(targets)
            
            if len(np.unique(y)) < 2:
                return False
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.difficulty_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
            self.difficulty_predictor.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.difficulty_predictor.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model
            with open(self.difficulty_model_path, 'wb') as f:
                pickle.dump(self.difficulty_predictor, f)
            
            st.info(f"🎯 Difficulty predictor trained with {accuracy:.2f} accuracy!")
            return True
            
        except Exception as e:
            logging.error(f"Error training difficulty predictor: {e}")
            return False
    
    def predict_question_difficulty_for_user(self, question_data: Dict, user_stats: Dict) -> str:
        """
        Predict appropriate difficulty level for user using trained ML model
        """
        if self.difficulty_predictor is None:
            try:
                with open(self.difficulty_model_path, 'rb') as f:
                    self.difficulty_predictor = pickle.load(f)
            except:
                return "intermediate"  # Default
        
        try:
            # Create feature vector
            features = np.array([[
                user_stats.get('avg_response_time', 30),
                user_stats.get('avg_response_length', 50),
                1 if question_data.get('question_type') == 'multiple_choice' else 0,
                1 if question_data.get('question_type') == 'open_ended' else 0,
                1 if question_data.get('question_type') == 'coding' else 0,
            ]])
            
            prediction = self.difficulty_predictor.predict(features)[0]
            difficulty_map = {0: 'beginner', 1: 'intermediate', 2: 'advanced'}
            return difficulty_map.get(prediction, 'intermediate')
            
        except Exception as e:
            logging.error(f"Error predicting difficulty: {e}")
            return "intermediate"
    
    def analyze_response_patterns(self, user_responses: List[str]) -> Dict:
        """
        Analyze user response patterns using NLP
        """
        if not user_responses:
            return {}
        
        try:
            # Basic NLP analysis
            avg_length = np.mean([len(response.split()) for response in user_responses])
            avg_char_length = np.mean([len(response) for response in user_responses])
            
            # Vocabulary richness
            all_words = []
            for response in user_responses:
                all_words.extend(response.lower().split())
            
            unique_words = len(set(all_words))
            total_words = len(all_words)
            vocab_richness = unique_words / total_words if total_words > 0 else 0
            
            # Technical term usage (simple keyword matching)
            technical_terms = [
                'model', 'training', 'data', 'algorithm', 'neural', 'learning',
                'accuracy', 'loss', 'gradient', 'optimization', 'regularization',
                'overfitting', 'embedding', 'transformer', 'attention'
            ]
            
            technical_usage = 0
            for response in user_responses:
                for term in technical_terms:
                    if term in response.lower():
                        technical_usage += 1
                        break
            
            technical_usage_rate = technical_usage / len(user_responses)
            
            return {
                'avg_word_length': avg_length,
                'avg_char_length': avg_char_length,
                'vocabulary_richness': vocab_richness,
                'technical_usage_rate': technical_usage_rate,
                'response_count': len(user_responses)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing response patterns: {e}")
            return {}
    
    def get_personalized_recommendations(self, current_question: str, user_history: List[Dict], k: int = 3) -> List[Dict]:
        """
        Get personalized question recommendations using multiple ML techniques
        """
        recommendations = []
        
        # 1. Semantic similarity recommendations
        similar_questions = self.find_similar_questions(current_question, k)
        for q in similar_questions:
            q['recommendation_type'] = 'Similar Topic'
            recommendations.append(q)
        
        # 2. Weak area recommendations
        weak_area_questions = self.recommend_questions_for_weak_areas(user_history, k)
        for q in weak_area_questions:
            q['recommendation_type'] = 'Improvement Area'
            recommendations.append(q)
        
        # Remove duplicates and limit results
        seen_ids = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec.get('id') not in seen_ids:
                seen_ids.add(rec.get('id'))
                unique_recommendations.append(rec)
        
        return unique_recommendations[:k]