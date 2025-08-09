import openai
import json
import time
from typing import Dict, List, Optional, Tuple
import random

class OpenAIClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = openai.OpenAI(api_key=api_key)
        
    def generate_question(self, 
                         topic: str, 
                         difficulty: str, 
                         question_type: str = "multiple_choice") -> Dict:
        """
        Generate a new question using OpenAI API
        
        Args:
            topic: The ML topic (e.g., "RAG", "Fine-tuning")
            difficulty: "beginner", "intermediate", or "advanced"  
            question_type: "multiple_choice", "open_ended", or "coding"
        
        Returns:
            Dictionary containing the generated question
        """
        try:
            prompt = self._build_question_prompt(topic, difficulty, question_type)
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert Machine Learning Engineer creating interview questions. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse the JSON response
            question_data = json.loads(response.choices[0].message.content)
            
            # Add metadata
            question_data["id"] = f"{topic.lower()}_{question_type}_{int(time.time())}"
            question_data["topic"] = topic
            question_data["difficulty"] = difficulty
            question_data["question_type"] = question_type
            question_data["generated"] = True
            
            return question_data
            
        except Exception as e:
            return self._get_fallback_question(topic, difficulty, question_type, str(e))
    
    def evaluate_answer(self, 
                       question: str, 
                       user_answer: str, 
                       correct_answer: Optional[str] = None) -> Dict:
        """
        Evaluate an open-ended answer using OpenAI
        
        Args:
            question: The original question
            user_answer: The user's response
            correct_answer: Expected answer (if available)
        
        Returns:
            Dictionary containing evaluation results
        """
        try:
            prompt = self._build_evaluation_prompt(question, user_answer, correct_answer)
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert Machine Learning Engineer evaluating interview responses. Provide constructive feedback."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            # Parse the JSON response
            evaluation = json.loads(response.choices[0].message.content)
            return evaluation
            
        except Exception as e:
            return {
                "score": 5,  # Middle score as fallback
                "max_score": 10,
                "feedback": f"Unable to evaluate response: {str(e)}",
                "strengths": ["Response provided"],
                "improvements": ["Try to be more specific"],
                "error": True
            }
    
    def get_hint(self, question: str, user_answer: str) -> str:
        """
        Generate a helpful hint for a question
        """
        try:
            prompt = f"""
            Question: {question}
            User's current answer: {user_answer}
            
            Provide a helpful hint to guide the user toward the correct answer without giving it away completely.
            Keep the hint concise and encouraging.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful ML tutor providing hints."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Try to think about the core concepts related to this topic. ({str(e)})"
    
    def _build_question_prompt(self, topic: str, difficulty: str, question_type: str) -> str:
        """Build the prompt for question generation"""
        
        if question_type == "multiple_choice":
            return f"""
            Generate a {difficulty} level multiple choice question about {topic} in Machine Learning Engineering.
            
            Return your response as JSON with this exact structure:
            {{
                "question": "Your question here",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer": 0,
                "explanation": "Detailed explanation of why this is correct",
                "difficulty": "{difficulty}"
            }}
            
            Make sure the question is practical and relevant to ML engineering interviews.
            Include plausible distractors in the wrong options.
            """
            
        elif question_type == "open_ended":
            return f"""
            Generate a {difficulty} level open-ended question about {topic} in Machine Learning Engineering.
            
            Return your response as JSON with this exact structure:
            {{
                "question": "Your thought-provoking question here",
                "difficulty": "{difficulty}",
                "topics": ["relevant", "keywords", "here"],
                "sample_answer_points": ["Key point 1", "Key point 2", "Key point 3"]
            }}
            
            Make sure the question requires deep understanding and practical knowledge.
            """
            
        elif question_type == "coding":
            return f"""
            Generate a {difficulty} level coding challenge about {topic} in Machine Learning Engineering.
            
            Return your response as JSON with this exact structure:
            {{
                "question": "Clear problem description",
                "difficulty": "{difficulty}",
                "starter_code": "# Starter code template with function signature",
                "topics": ["relevant", "topics"],
                "hints": ["Hint 1", "Hint 2"]
            }}
            
            Make sure the coding challenge is practical and tests ML engineering skills.
            """
            
        return "Generate a machine learning engineering question."
    
    def _build_evaluation_prompt(self, question: str, user_answer: str, correct_answer: Optional[str]) -> str:
        """Build the prompt for answer evaluation"""
        
        base_prompt = f"""
        Question: {question}
        User's Answer: {user_answer}
        """
        
        if correct_answer:
            base_prompt += f"Expected Answer: {correct_answer}\n"
        
        base_prompt += """
        Evaluate this response on a scale of 1-10 and provide constructive feedback.
        
        Return your response as JSON with this exact structure:
        {
            "score": 8,
            "max_score": 10,
            "feedback": "Overall assessment of the response",
            "strengths": ["What they did well", "Another strength"],
            "improvements": ["Specific area to improve", "Another suggestion"],
            "accuracy": "How factually correct the response is"
        }
        
        Be fair but thorough in your evaluation. Focus on technical accuracy and completeness.
        """
        
        return base_prompt
    
    def _get_fallback_question(self, topic: str, difficulty: str, question_type: str, error: str) -> Dict:
        """Return a fallback question when API fails"""
        
        fallback_questions = {
            "multiple_choice": {
                "question": f"What is a key consideration when working with {topic} in machine learning?",
                "options": [
                    "Data quality and preprocessing",
                    "Model complexity only", 
                    "Hardware requirements only",
                    "None of the above"
                ],
                "correct_answer": 0,
                "explanation": "Data quality and preprocessing are fundamental considerations in any ML approach.",
                "difficulty": difficulty
            },
            "open_ended": {
                "question": f"Describe the main challenges and considerations when implementing {topic} in a production ML system.",
                "difficulty": difficulty,
                "topics": [topic.lower(), "production", "challenges"]
            },
            "coding": {
                "question": f"Write a function that demonstrates a key concept in {topic}.",
                "difficulty": difficulty,
                "starter_code": f"def {topic.lower()}_function():\n    # Implement here\n    pass",
                "topics": [topic.lower()]
            }
        }
        
        fallback = fallback_questions.get(question_type, fallback_questions["multiple_choice"])
        fallback["id"] = f"fallback_{topic.lower()}_{question_type}_{int(time.time())}"
        fallback["topic"] = topic
        fallback["question_type"] = question_type
        fallback["generated"] = False
        fallback["error"] = error
        
        return fallback