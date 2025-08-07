import streamlit as st
import os
from dotenv import load_dotenv
import json
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import time

# Import our custom modules
from openai_client import OpenAIClient
from question_types import QuestionManager, MultipleChoiceHandler, OpenEndedHandler, CodingHandler
from evaluator import PerformanceEvaluator, QuestionAttempt, AdaptiveDifficultyManager
from database import DatabaseManager
from ml_engine import MLEngine

# Explicitly load environment variables
load_dotenv()
print(f"Debug at startup - .env loaded. API key exists: {bool(os.getenv('OPENAI_API_KEY'))}")

st.set_page_config(
    page_title="ML Engineer Interview Trainer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def init_components():
    """Initialize database, question manager, and ML engine (cached)"""
    db = DatabaseManager()
    question_manager = QuestionManager()
    ml_engine = MLEngine()
    return db, question_manager, ml_engine

def initialize_session_state():
    """Initialize all session state variables"""
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    if 'current_question_data' not in st.session_state:
        st.session_state.current_question_data = None
    if 'question_start_time' not in st.session_state:
        st.session_state.question_start_time = None
    if 'openai_api_key' not in st.session_state:
        # Load API key only from environment/.env file
        api_key = os.getenv('OPENAI_API_KEY', '')
        
        # If not found, try loading .env again (Streamlit might need explicit reload)
        if not api_key:
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY', '')
        
        # If still not found, try reading from .env file directly
        if not api_key:
            try:
                with open('.env', 'r') as f:
                    for line in f:
                        if line.startswith('OPENAI_API_KEY='):
                            api_key = line.split('=', 1)[1].strip()
                            break
            except FileNotFoundError:
                pass
        
        st.session_state.openai_api_key = api_key
        print(f"Debug: API key loaded: {bool(api_key)} (length: {len(api_key) if api_key else 0})")
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = PerformanceEvaluator()
    if 'difficulty_manager' not in st.session_state:
        st.session_state.difficulty_manager = AdaptiveDifficultyManager()
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    if 'awaiting_answer' not in st.session_state:
        st.session_state.awaiting_answer = False
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    if 'api_question_count' not in st.session_state:
        st.session_state.api_question_count = 0
    if 'bank_question_count' not in st.session_state:
        st.session_state.bank_question_count = 0
    if 'selected_tab' not in st.session_state:
        st.session_state.selected_tab = "🏠 Home"
    if 'show_api_input' not in st.session_state:
        st.session_state.show_api_input = False

def get_openai_client():
    """Get OpenAI client with minimal complexity"""
    if not st.session_state.openai_api_key:
        return None
        
    # Use the OpenAIClient wrapper that handles all the complexity
    try:
        from openai_client import OpenAIClient
        return OpenAIClient(st.session_state.openai_api_key)
    except Exception as e:
        # Only show error in debug mode
        print(f"Debug: OpenAI client creation failed: {str(e)}")
        return None

def main():
    initialize_session_state()
    db, question_manager, ml_engine = init_components()
    
    st.title("🤖 ML Engineer Interview Trainer")
    st.markdown("### Train and prep for Machine Learning Engineering interviews with AI-generated questions")
    
    # Sidebar for navigation and stats
    with st.sidebar:
        st.header("Navigation")
        
        
        tab = st.selectbox(
            "Choose Section:",
            ["🏠 Home", "📚 Practice Questions", "📊 Progress Dashboard", "🏆 Achievements", "⚙️ Settings"],
            index=["🏠 Home", "📚 Practice Questions", "📊 Progress Dashboard", "🏆 Achievements", "⚙️ Settings"].index(st.session_state.selected_tab),
            key="nav_selectbox"
        )
        
        # Update session state when user manually changes selection
        if tab != st.session_state.selected_tab:
            st.session_state.selected_tab = tab
        
        st.divider()
        
        # Current session stats
        session_stats = st.session_state.evaluator.get_session_stats()
        if session_stats.total_questions > 0:
            st.subheader("📈 Session Stats")
            st.metric("Questions Answered", session_stats.total_questions)
            st.metric("Session Accuracy", f"{session_stats.accuracy:.1f}%")
            st.metric("Current Streak", st.session_state.evaluator.streak_count)
            
            if session_stats.categories_covered:
                st.write("**Topics Practiced:**")
                for cat in session_stats.categories_covered:
                    st.write(f"• {cat}")
        
        # Overall stats from database
        st.divider()
        try:
            total_stats = db.get_total_stats()
            if total_stats["total_questions"] > 0:
                st.subheader("🎯 Overall Stats")
                st.metric("Total Questions", total_stats["total_questions"])
                st.metric("Overall Accuracy", f"{total_stats['overall_accuracy']:.1f}%")
                st.metric("Study Time", f"{total_stats['total_study_time_minutes']:.0f} min")
        except:
            pass  # Database might not be initialized yet
    
    # Main content based on selected tab
    if tab == "🏠 Home":
        show_home_page(db)
    elif tab == "📚 Practice Questions":
        show_practice_page(db, question_manager, ml_engine)
    elif tab == "📊 Progress Dashboard":
        show_progress_page(db)
    elif tab == "🏆 Achievements":
        show_achievements_page(db)
    elif tab == "⚙️ Settings":
        show_settings_page(db, ml_engine)

def show_home_page(db):
    st.header("Welcome to ML Engineer Interview Training!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Available Topics")
        topics = [
            "📖 Definitions (LLM, Encoder, Fine-tuning, Tokens, etc.)",
            "🎯 ML Fundamentals (Supervised/Unsupervised, Overfitting, etc.)",
            "🧠 Deep Learning Basics (Neural Networks, Backprop, etc.)",
            "🔍 RAG (Retrieval Augmented Generation)",
            "🎯 Fine-tuning & Transfer Learning", 
            "🏗️ Model Architecture & Design",
            "📊 Data Processing & Feature Engineering",
            "🔄 MLOps & Model Deployment",
            "📈 Model Evaluation & Metrics",
            "💾 Vector Databases & Embeddings",
            "🧠 Transformer Architecture"
        ]
        
        for topic in topics:
            st.markdown(f"• {topic}")
    
    with col2:
        st.subheader("🎯 Question Types")
        st.markdown("""
        • **Multiple Choice**: Quick concept checks
        • **Open-ended Theory**: Deep understanding questions  
        • **Coding Challenges**: Practical implementation tasks
        """)
        
        st.subheader("✨ Features")
        st.markdown("""
        • **AI-generated questions** using OpenAI
        • **ML-powered recommendations** using semantic similarity
        • **Smart difficulty prediction** with trained ML models
        • **Personalized feedback** and explanations
        • **Vector embeddings** for question similarity search
        • **Progress tracking** and weak area identification
        • **Achievement system** and streaks
        """)
        
        st.subheader("🤖 ML Components")
        st.markdown("""
        • **Sentence Transformers** for question embeddings
        • **FAISS vector database** for similarity search
        • **Random Forest models** for difficulty prediction
        • **NLP analysis** of response patterns
        • **Recommendation engine** for personalized learning
        """)
    
    # Recent achievements
    try:
        recent_achievements = db.get_achievements(limit=3)
        if recent_achievements:
            st.subheader("🏆 Recent Achievements")
            for achievement in recent_achievements:
                st.success(f"🎉 {achievement['title']}: {achievement['description']}")
    except:
        pass
    
    if st.button("🚀 Start Practice Session", type="primary", use_container_width=True):
        if not st.session_state.current_session_id:
            st.session_state.current_session_id = db.create_session()
        # Navigate to Practice Questions page
        st.session_state.selected_tab = "📚 Practice Questions"
        st.rerun()

def show_practice_page(db, question_manager, ml_engine):
    st.header("📚 Practice Questions")
    
    # Initialize session if needed
    if not st.session_state.current_session_id:
        st.session_state.current_session_id = db.create_session()
    
    # Configuration row
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_topic = st.selectbox(
            "Select Topic:",
            ["Definitions", "ML Fundamentals", "Deep Learning Basics", "RAG", "Fine-tuning", 
             "Model Architecture", "Data Processing", "MLOps", "Evaluation", 
             "Vector Databases", "Transformers"]
        )
    
    with col2:
        question_type = st.selectbox(
            "Question Type:",
            ["multiple_choice", "open_ended", "coding"]
        )
    
    with col3:
        # Get adaptive difficulty or let user choose
        adaptive_difficulty = st.session_state.difficulty_manager.get_difficulty(selected_topic)
        difficulty = st.selectbox(
            "Difficulty:",
            ["beginner", "intermediate", "advanced"],
            index=["beginner", "intermediate", "advanced"].index(adaptive_difficulty)
        )
    
    # Smart question generation with API usage limiting
    def get_next_question_source():
        """Determine whether to use AI or bank question based on usage patterns"""
        
        # Check if we have API key (without initializing client)
        if not st.session_state.openai_api_key:
            return "bank"
        
        # Strategy: Alternate between AI and bank questions
        # If we've used AI recently, prefer bank questions
        total_questions = st.session_state.api_question_count + st.session_state.bank_question_count
        
        # Use AI for every 3rd question to limit API usage
        if total_questions % 3 == 0:
            return "ai"
        else:
            return "bank"
    
    # Question generation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🎲 Get Smart Question", type="primary"):
            source = get_next_question_source()
            
            if source == "ai":
                openai_client = get_openai_client()
                if openai_client:
                    with st.spinner("🤖 Generating AI question..."):
                        question_data = openai_client.generate_question(
                            selected_topic, difficulty, question_type
                        )
                        st.session_state.api_question_count += 1
                        st.session_state.current_question_data = question_data
                        st.session_state.current_question = question_data
                        st.session_state.question_start_time = time.time()
                        st.session_state.awaiting_answer = True
                        st.session_state.last_result = None
                        st.info(f"🤖 AI-generated question (API usage: {st.session_state.api_question_count})")
                    st.rerun()
                else:
                    # Fallback to bank question
                    source = "bank"
            
            if source == "bank":
                question_data = question_manager.get_random_question(
                    selected_topic, question_type, difficulty
                )
                if question_data:
                    st.session_state.bank_question_count += 1
                    st.session_state.current_question_data = question_data
                    st.session_state.current_question = question_data
                    st.session_state.question_start_time = time.time()
                    st.session_state.awaiting_answer = True
                    st.session_state.last_result = None
                    st.info(f"📚 Bank question (Bank usage: {st.session_state.bank_question_count})")
                    st.rerun()
                else:
                    st.warning("No questions found for this combination. Try different settings.")
    
    with col2:
        if st.button("🤖 Force AI Question"):
            openai_client = get_openai_client()
            
            if openai_client:
                with st.spinner("Generating question with AI..."):
                    question_data = openai_client.generate_question(
                        selected_topic, difficulty, question_type
                    )
                    st.session_state.api_question_count += 1
                    st.session_state.current_question_data = question_data
                    st.session_state.current_question = question_data
                    st.session_state.question_start_time = time.time()
                    st.session_state.awaiting_answer = True
                    st.session_state.last_result = None
                st.rerun()
            else:
                st.error("OpenAI not available. Please check your API key in Settings.")
    
    with col3:
        if st.button("📚 Force Bank Question"):
            question_data = question_manager.get_random_question(
                selected_topic, question_type, difficulty
            )
            if question_data:
                st.session_state.bank_question_count += 1
                st.session_state.current_question_data = question_data
                st.session_state.current_question = question_data
                st.session_state.question_start_time = time.time()
                st.session_state.awaiting_answer = True
                st.session_state.last_result = None
                st.rerun()
            else:
                st.warning("No questions found for this combination. Try different settings.")
    
    # Show API usage stats
    if 'api_question_count' in st.session_state and 'bank_question_count' in st.session_state:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🤖 AI Questions", st.session_state.api_question_count)
        with col2:
            st.metric("📚 Bank Questions", st.session_state.bank_question_count)
        with col3:
            total = st.session_state.api_question_count + st.session_state.bank_question_count
            if total > 0:
                api_percentage = (st.session_state.api_question_count / total) * 100
                st.metric("API Usage %", f"{api_percentage:.1f}%")
    
    st.divider()
    
    # Display current question
    if st.session_state.current_question:
        display_question_and_handle_response(db, question_manager, selected_topic, ml_engine)
    else:
        st.info("👆 Click one of the buttons above to get your first question!")
        
        # ML-powered recommendations
        show_ml_recommendations(db, ml_engine)
        
        # Show sample questions available
        st.subheader("📖 Sample Topics Available")
        bank = question_manager.load_question_bank()
        for category, questions in bank.get("categories", {}).items():
            total = sum(len(q) for q in questions.values())
            st.write(f"• **{category}**: {total} questions")

def display_question_and_handle_response(db, question_manager, selected_topic, ml_engine):
    """Display question and handle user response"""
    
    question_data = st.session_state.current_question_data
    
    # Only get OpenAI client if needed (for evaluation)
    try:
        openai_client = get_openai_client()
    except:
        openai_client = None
    
    # Get appropriate handler
    handler = question_manager.get_handler(
        question_data.get('question_type', 'multiple_choice'), 
        openai_client
    )
    
    # Show last result if available
    if st.session_state.last_result:
        result = st.session_state.last_result
        
        if result['correct']:
            st.success(f"✅ {result['performance_emoji']} {result['performance_level']}! Score: {result['score']:.1f}/{result['max_score']}")
        else:
            st.error(f"❌ {result['performance_emoji']} {result['performance_level']}. Score: {result['score']:.1f}/{result['max_score']}")
        
        st.info(f"⏱️ Time taken: {result['time_taken']:.1f}s | Streak: {result['streak']}")
        
        with st.expander("📝 Detailed Feedback"):
            st.write(result['feedback_message'])
        
        st.divider()
    
    # Render the question
    if st.session_state.awaiting_answer:
        st.subheader(f"Question ({question_data.get('difficulty', 'unknown').title()})")
        
        user_response = handler.render_question(question_data)
        
        # Submit button
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("Submit Answer", type="primary", disabled=user_response is None):
                if user_response is not None:
                    process_answer(db, handler, question_data, user_response, selected_topic)
                    st.rerun()
        
        with col2:
            if st.button("Skip Question"):
                st.session_state.current_question = None
                st.session_state.current_question_data = None
                st.session_state.awaiting_answer = False
                st.rerun()
        
        with col3:
            if question_data.get('question_type') == 'open_ended' and openai_client:
                if st.button("💡 Get Hint"):
                    if user_response:
                        hint = openai_client.get_hint(question_data['question'], user_response)
                        st.info(f"💡 Hint: {hint}")
    else:
        # Question answered, show next question button
        if st.button("➡️ Next Question", type="primary"):
            st.session_state.current_question = None
            st.session_state.current_question_data = None
            st.session_state.awaiting_answer = False
            st.session_state.last_result = None
            st.rerun()

def process_answer(db, handler, question_data, user_response, selected_topic):
    """Process the user's answer and record results"""
    
    # Calculate time taken
    time_taken = time.time() - st.session_state.question_start_time
    
    # Evaluate the response
    evaluation = handler.evaluate_response(question_data, user_response)
    
    # Create attempt record
    attempt = QuestionAttempt(
        question_id=question_data.get('id', 'unknown'),
        category=selected_topic,
        question_type=question_data.get('question_type', 'multiple_choice'),
        difficulty=question_data.get('difficulty', 'beginner'),
        user_answer=str(user_response),
        correct=evaluation.get('correct', False),
        score=evaluation.get('score', 0),
        max_score=evaluation.get('max_score', 1),
        time_taken=time_taken,
        timestamp=datetime.now(),
        feedback=evaluation.get('feedback', ''),
        ai_evaluated=evaluation.get('ai_evaluated', False)
    )
    
    # Record in evaluator and get feedback
    feedback = st.session_state.evaluator.record_attempt(attempt)
    
    # Store in database
    db.record_attempt(st.session_state.current_session_id, attempt, question_data)
    
    # Update adaptive difficulty
    st.session_state.difficulty_manager.update_difficulty(
        selected_topic, st.session_state.evaluator.current_session_attempts
    )
    
    # Check for achievements
    achievements = st.session_state.evaluator.get_achievements()
    for achievement in achievements:
        try:
            db.record_achievement(achievement, st.session_state.current_session_id)
        except:
            pass  # Avoid duplicate achievements
    
    # Store results for display
    st.session_state.last_result = feedback
    st.session_state.awaiting_answer = False

def show_progress_page(db):
    st.header("📊 Progress Dashboard")
    
    try:
        # Overall statistics
        total_stats = db.get_total_stats()
        
        if total_stats["total_questions"] == 0:
            st.info("No practice sessions yet. Start practicing to see your progress!")
            return
        
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Questions", total_stats["total_questions"])
        with col2:
            st.metric("Overall Accuracy", f"{total_stats['overall_accuracy']:.1f}%")
        with col3:
            st.metric("Total Sessions", total_stats["total_sessions"])
        with col4:
            st.metric("Study Time", f"{total_stats['total_study_time_minutes']:.0f} min")
        
        # Progress over time
        st.subheader("📈 Progress Over Time")
        progress_df = db.get_progress_over_time(days=30)
        
        if not progress_df.empty:
            fig = px.line(progress_df, x='date', y='avg_accuracy', 
                         title="Accuracy Trend (Last 30 Days)",
                         labels={'avg_accuracy': 'Accuracy (%)', 'date': 'Date'})
            fig.update_traces(line=dict(color='#1f77b4', width=3))
            st.plotly_chart(fig, use_container_width=True)
            
            # Questions per day
            fig2 = px.bar(progress_df, x='date', y='total_questions',
                         title="Questions Answered Per Day",
                         labels={'total_questions': 'Questions', 'date': 'Date'})
            fig2.update_traces(marker_color='#ff7f0e')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Not enough data for time-series charts yet. Keep practicing!")
        
        # Category performance
        st.subheader("🎯 Performance by Category")
        category_df = db.get_category_performance()
        
        if not category_df.empty:
            # Create category performance chart
            fig3 = px.bar(category_df, x='category', y='accuracy',
                         color='difficulty', barmode='group',
                         title="Accuracy by Category and Difficulty",
                         labels={'accuracy': 'Accuracy (%)', 'category': 'Category'})
            st.plotly_chart(fig3, use_container_width=True)
            
            # Show detailed table
            with st.expander("📋 Detailed Category Stats"):
                st.dataframe(category_df, use_container_width=True)
        
        # Weak areas
        st.subheader("🎯 Areas for Improvement")
        weak_areas = db.get_weak_areas_from_history()
        
        if weak_areas:
            for area in weak_areas:
                st.warning(f"**{area['category']}**: {area['accuracy']:.1f}% accuracy ({area['attempts']} attempts)")
        else:
            st.success("Great job! No major weak areas identified.")
            
    except Exception as e:
        st.error(f"Error loading progress data: {e}")

def show_achievements_page(db):
    st.header("🏆 Achievements")
    
    try:
        achievements = db.get_achievements()
        
        if not achievements:
            st.info("No achievements yet. Start practicing to earn your first achievement!")
            return
        
        # Group achievements by type
        achievement_types = {}
        for ach in achievements:
            ach_type = ach['achievement_type']
            if ach_type not in achievement_types:
                achievement_types[ach_type] = []
            achievement_types[ach_type].append(ach)
        
        for ach_type, achs in achievement_types.items():
            st.subheader(f"{ach_type.title()} Achievements")
            
            for ach in achs:
                earned_date = datetime.fromisoformat(ach['earned_date']).strftime("%Y-%m-%d %H:%M")
                st.success(f"🎉 **{ach['title']}** - {ach['description']} (Earned: {earned_date})")
            
            st.divider()
            
    except Exception as e:
        st.error(f"Error loading achievements: {e}")

def show_ml_recommendations(db, ml_engine):
    """Show ML-powered question recommendations"""
    st.subheader("🤖 AI-Powered Recommendations")
    
    # Initialize ML models if needed
    if st.button("🚀 Initialize ML Models", help="Build question embeddings for smart recommendations"):
        with st.spinner("Building question embeddings using ML..."):
            success = ml_engine.build_question_embeddings()
            if success:
                st.success("✅ ML models ready! You'll now get smart recommendations.")
            else:
                st.error("❌ Failed to initialize ML models")
        st.rerun()
    
    # Get user history for recommendations
    try:
        attempts_data = []
        if st.session_state.evaluator.current_session_attempts:
            for attempt in st.session_state.evaluator.current_session_attempts:
                attempts_data.append({
                    'category': attempt.category,
                    'correct': attempt.correct,
                    'time_taken': attempt.time_taken,
                    'difficulty': attempt.difficulty
                })
        
        if attempts_data:
            # Get weak area recommendations
            weak_recommendations = ml_engine.recommend_questions_for_weak_areas(attempts_data)
            
            if weak_recommendations:
                st.write("**📈 Recommended for improvement:**")
                for i, rec in enumerate(weak_recommendations[:3]):
                    with st.expander(f"💡 {rec['category']} - {rec['difficulty'].title()}"):
                        st.write(f"**Question:** {rec['question'][:100]}...")
                        st.write(f"**Reason:** {rec.get('recommendation_reason', 'Weak area')}")
                        if st.button(f"Practice This", key=f"rec_{i}"):
                            # Load this question
                            st.session_state.current_question_data = rec
                            st.session_state.current_question = rec
                            st.session_state.question_start_time = time.time()
                            st.session_state.awaiting_answer = True
                            st.session_state.last_result = None
                            st.rerun()
        else:
            st.info("Answer a few questions to get personalized ML-powered recommendations!")
            
    except Exception as e:
        st.info("ML recommendations will appear after you answer some questions.")

def show_settings_page(db, ml_engine):
    st.header("⚙️ Settings")
    
    # OpenAI API Key
    st.subheader("🔑 OpenAI API Configuration")
    
    # Show API key status
    if st.session_state.openai_api_key:
        st.success("✅ API key is configured from .env file")
        st.write(f"• API key length: {len(st.session_state.openai_api_key)}")
        st.write(f"• Key starts with: {st.session_state.openai_api_key[:15]}...")
    else:
        st.error("❌ No API key found in .env file")
        st.markdown("""
        **To configure your OpenAI API key:**
        1. Create or edit the `.env` file in your project directory
        2. Add: `OPENAI_API_KEY=your_api_key_here`
        3. Restart the application
        
        Get your API key from: https://platform.openai.com/api-keys
        """)
    
    st.divider()
    
    # User preferences
    st.subheader("🎛️ Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        default_difficulty = st.selectbox(
            "Default Difficulty:", 
            ["beginner", "intermediate", "advanced"],
            index=["beginner", "intermediate", "advanced"].index(
                db.get_preference("default_difficulty", "intermediate")
            )
        )
        
        questions_per_session = st.selectbox(
            "Questions per Session:", 
            [5, 10, 15, 20],
            index=[5, 10, 15, 20].index(
                int(db.get_preference("questions_per_session", "10"))
            )
        )
    
    with col2:
        enable_ai_feedback = st.checkbox(
            "Enable AI Feedback", 
            value=db.get_preference("enable_ai_feedback", "true") == "true"
        )
        show_explanations = st.checkbox(
            "Show Explanations", 
            value=db.get_preference("show_explanations", "true") == "true"
        )
    
    if st.button("Save Preferences"):
        db.set_preference("default_difficulty", default_difficulty)
        db.set_preference("questions_per_session", str(questions_per_session))
        db.set_preference("enable_ai_feedback", str(enable_ai_feedback).lower())
        db.set_preference("show_explanations", str(show_explanations).lower())
        st.success("✅ Preferences saved!")
    
    st.divider()
    
    # ML Engine Settings
    st.subheader("🤖 Machine Learning Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Build Question Embeddings"):
            with st.spinner("Building ML embeddings for smart recommendations..."):
                success = ml_engine.build_question_embeddings()
                if success:
                    st.success("✅ Question embeddings built successfully!")
                else:
                    st.error("❌ Failed to build embeddings")
        
        if st.button("🎯 Train Difficulty Predictor"):
            # Get user attempt data
            try:
                attempts = []
                for attempt in st.session_state.evaluator.current_session_attempts:
                    attempts.append({
                        'time_taken': attempt.time_taken,
                        'user_answer': attempt.user_answer,
                        'question_type': attempt.question_type,
                        'difficulty': attempt.difficulty,
                        'correct': attempt.correct
                    })
                
                if len(attempts) >= 10:
                    success = ml_engine.train_difficulty_predictor(attempts)
                    if success:
                        st.success("✅ Difficulty predictor trained!")
                    else:
                        st.warning("⚠️ Need more diverse data for training")
                else:
                    st.info("📊 Need at least 10 attempts to train the model")
                    
            except Exception as e:
                st.error(f"❌ Training failed: {e}")
    
    with col2:
        # Show ML model status
        st.write("**🔍 ML Model Status:**")
        
        # Check if embeddings exist
        embeddings_exist = ml_engine.faiss_index is not None
        st.write(f"• Question Embeddings: {'✅ Ready' if embeddings_exist else '❌ Not built'}")
        
        # Check if difficulty predictor exists
        predictor_exists = ml_engine.difficulty_predictor is not None
        st.write(f"• Difficulty Predictor: {'✅ Trained' if predictor_exists else '❌ Not trained'}")
        
        # Show recommendation stats
        if embeddings_exist and st.session_state.evaluator.current_session_attempts:
            st.write(f"• Session Questions: {len(st.session_state.evaluator.current_session_attempts)}")
            
            # Show similarity search example
            if st.button("🔍 Test Similarity Search"):
                test_query = "What is overfitting in machine learning?"
                similar = ml_engine.find_similar_questions(test_query, k=3)
                st.write("**Similar questions found:**")
                for i, q in enumerate(similar[:2], 1):
                    st.write(f"{i}. {q.get('question', '')[:80]}... (Score: {q.get('similarity_score', 0):.2f})")
    
    st.divider()
    
    # Data management
    st.subheader("📊 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Progress Data"):
            try:
                export_file = f"ml_trainer_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                db.export_data(export_file)
                st.success(f"✅ Data exported to {export_file}")
            except Exception as e:
                st.error(f"❌ Export failed: {e}")
        
        if st.button("🔄 End Current Session"):
            if st.session_state.current_session_id:
                session_stats = st.session_state.evaluator.get_session_stats()
                db.update_session(st.session_state.current_session_id, session_stats)
                st.session_state.current_session_id = None
                st.session_state.evaluator.reset_session()
                st.success("✅ Session ended and saved!")
    
    with col2:
        if st.button("🗑️ Reset Session Progress", type="secondary"):
            st.session_state.evaluator.reset_session()
            st.session_state.current_session_id = None
            st.success("✅ Session progress reset!")
        
        # Danger zone
        if st.button("⚠️ Reset All Data", type="secondary"):
            if st.checkbox("I understand this will delete all my progress"):
                if st.button("🗑️ Confirm Delete All Data"):
                    try:
                        os.remove(db.db_path)
                        st.success("✅ All data deleted. Restart the app to begin fresh.")
                        st.stop()
                    except Exception as e:
                        st.error(f"❌ Failed to delete data: {e}")

if __name__ == "__main__":
    main()