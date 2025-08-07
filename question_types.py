import streamlit as st
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from openai_client import OpenAIClient

class QuestionHandler:
    """Base class for handling different question types"""
    
    def __init__(self, openai_client: Optional[OpenAIClient] = None):
        self.openai_client = openai_client
    
    def render_question(self, question_data: Dict) -> Any:
        """Render the question UI and return user response"""
        raise NotImplementedError
    
    def evaluate_response(self, question_data: Dict, user_response: Any) -> Dict:
        """Evaluate user response and return results"""
        raise NotImplementedError

class MultipleChoiceHandler(QuestionHandler):
    """Handler for multiple choice questions"""
    
    def render_question(self, question_data: Dict) -> Optional[str]:
        """Render multiple choice question and return selected option"""
        
        st.markdown(f"**Question ({question_data.get('difficulty', 'unknown').title()}):**")
        st.markdown(question_data['question'])
        
        # Display options
        options = question_data.get('options', [])
        if not options:
            st.error("No options provided for this question")
            return None
        
        # Create radio button for selection with stable key
        question_id = question_data.get('id', 'unknown')
        selected = st.radio(
            "Choose your answer:",
            options,
            key=f"mc_{question_id}",
            index=None  # Start with no selection
        )
        
        return selected
    
    def evaluate_response(self, question_data: Dict, user_response: str) -> Dict:
        """Evaluate multiple choice response"""
        
        if user_response is None:
            return {
                "correct": False,
                "score": 0,
                "feedback": "No answer selected",
                "explanation": question_data.get('explanation', ''),
                "user_answer": None,
                "correct_answer": question_data.get('options', [])[question_data.get('correct_answer', 0)] if question_data.get('options') else None
            }
        
        options = question_data.get('options', [])
        correct_index = question_data.get('correct_answer', 0)
        correct_answer = options[correct_index] if correct_index < len(options) else None
        
        is_correct = user_response == correct_answer
        
        return {
            "correct": is_correct,
            "score": 1 if is_correct else 0,
            "feedback": "Correct!" if is_correct else "Incorrect. " + question_data.get('explanation', ''),
            "explanation": question_data.get('explanation', ''),
            "user_answer": user_response,
            "correct_answer": correct_answer
        }

class OpenEndedHandler(QuestionHandler):
    """Handler for open-ended questions"""
    
    def render_question(self, question_data: Dict) -> Optional[str]:
        """Render open-ended question and return user response"""
        
        st.markdown(f"**Question ({question_data.get('difficulty', 'unknown').title()}):**")
        st.markdown(question_data['question'])
        
        # Show relevant topics if available
        topics = question_data.get('topics', [])
        if topics:
            st.markdown("**Related topics:** " + ", ".join(topics))
        
        # Text area for response
        question_id = question_data.get('id', 'unknown')
        response = st.text_area(
            "Your answer:",
            height=200,
            key=f"oe_{question_id}",
            placeholder="Provide a detailed explanation..."
        )
        
        return response.strip() if response else None
    
    def evaluate_response(self, question_data: Dict, user_response: str) -> Dict:
        """Evaluate open-ended response using AI or fallback criteria"""
        
        if not user_response:
            return {
                "score": 0,
                "max_score": 10,
                "feedback": "No response provided",
                "strengths": [],
                "improvements": ["Provide a response to the question"],
                "ai_evaluated": False
            }
        
        # Try AI evaluation if client is available
        if self.openai_client:
            try:
                ai_evaluation = self.openai_client.evaluate_answer(
                    question_data['question'], 
                    user_response
                )
                ai_evaluation["ai_evaluated"] = True
                return ai_evaluation
            except Exception as e:
                st.warning(f"AI evaluation failed: {e}")
        
        # Fallback evaluation
        return self._fallback_evaluation(question_data, user_response)
    
    def _fallback_evaluation(self, question_data: Dict, user_response: str) -> Dict:
        """Simple rule-based evaluation when AI is unavailable"""
        
        response_length = len(user_response.split())
        topics = question_data.get('topics', [])
        
        # Basic scoring based on length and keyword matching
        base_score = min(3, response_length // 10)  # 1 point per 10 words, max 3
        
        keyword_score = 0
        for topic in topics:
            if topic.lower() in user_response.lower():
                keyword_score += 1
        
        total_score = min(10, base_score + keyword_score * 2)
        
        feedback_parts = []
        if response_length < 20:
            feedback_parts.append("Try to provide more detailed explanations")
        if keyword_score == 0 and topics:
            feedback_parts.append(f"Consider mentioning: {', '.join(topics)}")
        if response_length > 50:
            feedback_parts.append("Good detailed response")
        
        return {
            "score": total_score,
            "max_score": 10,
            "feedback": ". ".join(feedback_parts) if feedback_parts else "Response received",
            "strengths": ["Response provided"] + (["Good length"] if response_length > 30 else []),
            "improvements": feedback_parts if feedback_parts else ["Keep up the good work"],
            "ai_evaluated": False
        }

class CodingHandler(QuestionHandler):
    """Handler for coding questions"""
    
    def render_question(self, question_data: Dict) -> Optional[str]:
        """Render coding question and return user code"""
        
        st.markdown(f"**Coding Challenge ({question_data.get('difficulty', 'unknown').title()}):**")
        st.markdown(question_data['question'])
        
        # Show starter code if available
        starter_code = question_data.get('starter_code', '')
        if starter_code:
            st.markdown("**Starter code:**")
            st.code(starter_code, language="python")
        
        # Show hints if available
        hints = question_data.get('hints', [])
        if hints:
            with st.expander("💡 Hints"):
                for i, hint in enumerate(hints, 1):
                    st.markdown(f"{i}. {hint}")
        
        # Code editor
        question_id = question_data.get('id', 'unknown')
        user_code = st.text_area(
            "Your solution:",
            value=starter_code,
            height=300,
            key=f"code_{question_id}"
        )
        
        return user_code.strip() if user_code else None
    
    def evaluate_response(self, question_data: Dict, user_response: str) -> Dict:
        """Evaluate coding response"""
        
        if not user_response:
            return {
                "score": 0,
                "max_score": 10,
                "feedback": "No code provided",
                "syntax_check": "No code to check",
                "suggestions": ["Provide a solution to the coding challenge"],
                "ai_evaluated": False
            }
        
        # Basic syntax check
        syntax_feedback = self._check_python_syntax(user_response)
        
        # Try AI evaluation if available
        if self.openai_client:
            try:
                ai_evaluation = self._ai_code_evaluation(question_data, user_response)
                ai_evaluation["syntax_check"] = syntax_feedback
                ai_evaluation["ai_evaluated"] = True
                return ai_evaluation
            except Exception as e:
                st.warning(f"AI evaluation failed: {e}")
        
        # Fallback evaluation
        return self._fallback_code_evaluation(question_data, user_response, syntax_feedback)
    
    def _check_python_syntax(self, code: str) -> str:
        """Basic Python syntax checking"""
        try:
            compile(code, '<string>', 'exec')
            return "✅ Syntax appears valid"
        except SyntaxError as e:
            return f"❌ Syntax error: {e.msg} at line {e.lineno}"
        except Exception as e:
            return f"⚠️ Compilation issue: {str(e)}"
    
    def _ai_code_evaluation(self, question_data: Dict, user_code: str) -> Dict:
        """Use AI to evaluate code quality and correctness"""
        
        prompt = f"""
        Question: {question_data['question']}
        User's Code Solution:
        ```python
        {user_code}
        ```
        
        Evaluate this code solution and provide feedback.
        Return JSON with this structure:
        {{
            "score": 8,
            "max_score": 10,
            "feedback": "Overall assessment",
            "correctness": "How well it solves the problem",
            "code_quality": "Code style and best practices",
            "suggestions": ["Improvement 1", "Improvement 2"]
        }}
        """
        
        response = self.openai_client.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a senior ML engineer reviewing code. Provide constructive feedback."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        return json.loads(response.choices[0].message.content)
    
    def _fallback_code_evaluation(self, question_data: Dict, user_code: str, syntax_feedback: str) -> Dict:
        """Simple rule-based code evaluation"""
        
        # Basic scoring criteria
        score = 5  # Base score
        feedback_parts = []
        
        # Check for common good practices
        if "def " in user_code:
            score += 1
            feedback_parts.append("Good use of functions")
        
        if any(word in user_code.lower() for word in ["import", "from"]):
            score += 1
            feedback_parts.append("Appropriate library usage")
        
        if len(user_code.split('\n')) > 3:
            score += 1
            feedback_parts.append("Substantial implementation")
        
        # Check for basic error handling
        if "try:" in user_code or "except:" in user_code:
            score += 1
            feedback_parts.append("Good error handling")
        
        # Check for comments/documentation
        if "#" in user_code or '"""' in user_code:
            score += 1
            feedback_parts.append("Well documented")
        
        return {
            "score": min(10, score),
            "max_score": 10,
            "feedback": ". ".join(feedback_parts) if feedback_parts else "Code submitted",
            "correctness": "Basic implementation provided",
            "code_quality": syntax_feedback,
            "suggestions": ["Consider adding more comments", "Test edge cases"],
            "ai_evaluated": False
        }

class QuestionManager:
    """Manages loading and serving questions"""
    
    def __init__(self, question_bank_path: str = "question_bank.json"):
        self.question_bank_path = question_bank_path
        self.handlers = {
            "multiple_choice": MultipleChoiceHandler,
            "open_ended": OpenEndedHandler,
            "coding": CodingHandler
        }
        self._question_bank = None
    
    def load_question_bank(self) -> Dict:
        """Load questions from JSON file"""
        if self._question_bank is None:
            try:
                with open(self.question_bank_path, 'r') as f:
                    self._question_bank = json.load(f)
            except FileNotFoundError:
                st.error(f"Question bank file not found: {self.question_bank_path}")
                self._question_bank = {"categories": {}}
        return self._question_bank
    
    def get_questions_by_category(self, category: str) -> Dict:
        """Get all questions for a specific category"""
        bank = self.load_question_bank()
        return bank.get("categories", {}).get(category, {})
    
    def get_random_question(self, category: str, question_type: str, difficulty: Optional[str] = None) -> Optional[Dict]:
        """Get a random question matching criteria"""
        import random
        
        questions = self.get_questions_by_category(category).get(question_type, [])
        
        if difficulty:
            questions = [q for q in questions if q.get('difficulty') == difficulty]
        
        return random.choice(questions) if questions else None
    
    def get_handler(self, question_type: str, openai_client: Optional[OpenAIClient] = None) -> QuestionHandler:
        """Get appropriate handler for question type"""
        handler_class = self.handlers.get(question_type, MultipleChoiceHandler)
        return handler_class(openai_client)