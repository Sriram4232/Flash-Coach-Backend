from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timezone
from bson import ObjectId, json_util
import json
import os
import sys
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING
from groq import Groq
from coaching_orchestrator import (
    classify_problem, select_strategies, build_prompt, check_escalation_needed
)

# Add new imports for translation and voice
from deep_translator import GoogleTranslator
import speech_recognition as sr
from gtts import gTTS
import tempfile
import base64
import io
import re

translation_cache = {}

# Load environment variables
load_dotenv('.env')

app = Flask(__name__)
CORS(app)

# Configuration from environment
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
DATABASE_NAME = os.getenv('DATABASE_NAME', 'flash_coach')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
PORT = int(os.getenv('PORT', 5000))

print(f"Python version: {sys.version}")
print(f"Connecting to MongoDB: {DATABASE_NAME}")
print(f"Groq API: {'Configured' if GROQ_API_KEY else 'Not configured'}")

# Initialize MongoDB
try:
    client = MongoClient(
        MONGO_URI,
        tls=True,
        tlsAllowInvalidCertificates=True,
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=30000,
        socketTimeoutMS=30000
    )
    # Test connection
    client.admin.command('ping')
    print("✅ MongoDB connected successfully!")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    raise

db = client[DATABASE_NAME]

# Collections
teachers_collection = db['teachers']
classrooms_collection = db['classrooms']
strategies_collection = db['strategies']
interactions_collection = db['interactions']
escalations_collection = db['escalations']

# Initialize Groq client
groq_client = None
if GROQ_API_KEY:
    try:
        print("🔄 Initializing Groq client...")
        
        # Clean environment variables that might cause issues
        for key in ['proxies', 'PROXIES']:
            if key in os.environ:
                print(f"Removing env var: {key}")
                os.environ.pop(key, None)
        
        # Initialize with minimal parameters
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq client initialized successfully!")
        
        # Test the connection
        try:
            test_response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            print("✅ Groq API connection test passed!")
        except Exception as test_error:
            print(f"⚠️  Groq API test failed (but client created): {test_error}")
            
    except TypeError as e:
        print(f"❌ Groq client initialization failed (TypeError): {e}")
        print("Trying alternative initialization...")
        try:
            # Try alternative initialization method
            import groq as groq_module
            groq_client = groq_module.Groq(api_key=GROQ_API_KEY)
            print("✅ Groq client initialized via alternative method!")
        except Exception as e2:
            print(f"❌ Alternative method failed: {e2}")
            groq_client = None
    except Exception as e:
        print(f"❌ Groq client initialization failed: {e}")
        import traceback
        traceback.print_exc()
        groq_client = None
else:
    print("⚠️  Groq API key not found, using mock responses")

# Helper function to convert ObjectId to string
def serialize_doc(doc):
    """Convert MongoDB document to JSON-serializable format."""
    if doc is None:
        return None
    if '_id' in doc:
        doc['_id'] = str(doc['_id'])
    return doc

def serialize_docs(docs):
    """Convert list of MongoDB documents to JSON-serializable format."""
    return [serialize_doc(doc) for doc in docs]

# Initialize sample data
def init_db():
    """Initialize database with sample data if collections are empty."""
    
    # Use timezone-aware datetime
    now = datetime.now(timezone.utc)
    
    # Check if collections are empty
    if teachers_collection.count_documents({}) == 0:
        print("Creating sample teacher...")
        # Sample teacher
        teacher = {
            "name": "",
            "confidence_level": "medium",
            "years_experience": 5,
            "grade": "5th Grade",
            "created_at": now
        }
        teacher_id = teachers_collection.insert_one(teacher).inserted_id
        print(f"✅ Created teacher with ID: {teacher_id}")
        
        # Sample classroom
        classroom = {
            "teacher_id": str(teacher_id),
            "name": "Room 205",
            "grade_level": "5th Grade",
            "student_count": 24,
            "multi_level_flag": True,
            "created_at": now
        }
        classroom_id = classrooms_collection.insert_one(classroom).inserted_id
        print(f"✅ Created classroom with ID: {classroom_id}")
    
    if strategies_collection.count_documents({}) == 0:
        print("Creating sample strategies...")
        # Sample strategies
        strategies = [
            {
                "type": "classroom_management",
                "title": "Proximity Control",
                "description": "Move closer to disruptive students to redirect behavior non-verbally.",
                "difficulty": "simple",
                "created_at": now
            },
            {
                "type": "classroom_management",
                "title": "Positive Narration",
                "description": "Describe the positive behaviors you see from compliant students.",
                "difficulty": "simple",
                "created_at": now
            },
            {
                "type": "classroom_management",
                "title": "Behavior Momentum",
                "description": "Start with easy requests before making more challenging ones.",
                "difficulty": "moderate",
                "created_at": now
            },
            {
                "type": "conceptual_misunderstanding",
                "title": "Think-Pair-Share",
                "description": "Students think individually, discuss with a partner, then share with class.",
                "difficulty": "simple",
                "created_at": now
            },
            {
                "type": "conceptual_misunderstanding",
                "title": "Concrete Examples",
                "description": "Use physical objects or real-world examples to explain abstract concepts.",
                "difficulty": "moderate",
                "created_at": now
            },
            {
                "type": "multi_level_activity",
                "title": "Tiered Assignments",
                "description": "Create different versions of the same assignment at varying difficulty levels.",
                "difficulty": "moderate",
                "created_at": now
            },
            {
                "type": "assessment_anxiety",
                "title": "Practice Tests",
                "description": "Provide low-stakes practice tests to build confidence.",
                "difficulty": "simple",
                "created_at": now
            }
        ]
        result = strategies_collection.insert_many(strategies)
        print(f"✅ Created {len(result.inserted_ids)} strategies")
    
    # Create indexes
    interactions_collection.create_index([("created_at", DESCENDING)])
    interactions_collection.create_index([("teacher_id", ASCENDING)])
    strategies_collection.create_index([("type", ASCENDING)])
    
    print("✅ Database initialization complete!")

# Groq LLM Function - Using the OLD working prompt
def call_groq_llm(prompt: str):
    """Call Groq LLM API to get coaching advice."""
    if not groq_client:
        # Fallback to mock response if Groq is not configured
        print("⚠️  Using mock LLM response (Groq client not available)")
        
        # Enhanced mock response based on prompt content
        lower_prompt = prompt.lower()
        
        if "classroom management" in lower_prompt or "behavior" in lower_prompt or "disruptive" in lower_prompt:
            return {
                "strategies": [
                    {
                        "title": "Proximity Control",
                        "description": "Move closer to disruptive students without interrupting instruction. Your physical presence can redirect behavior."
                    },
                    {
                        "title": "Positive Narration",
                        "description": "Verbally acknowledge students who are following directions. 'I see Sarah is ready with her materials.'"
                    },
                    {
                        "title": "Clear Expectations",
                        "description": "Review classroom rules and procedures before starting the activity. Use visual reminders if needed."
                    }
                ],
                "summary": "For behavior management, focus on non-verbal cues, positive reinforcement, and establishing clear routines."
            }
        elif "understanding" in lower_prompt or "confused" in lower_prompt or "concept" in lower_prompt:
            return {
                "strategies": [
                    {
                        "title": "Think-Pair-Share",
                        "description": "Have students think individually for 1 minute, discuss with a partner for 2 minutes, then share with the class."
                    },
                    {
                        "title": "Concrete Examples",
                        "description": "Use real-world analogies or physical objects to make abstract concepts more tangible."
                    },
                    {
                        "title": "Exit Tickets",
                        "description": "Ask students to write one thing they learned and one question they still have at the end of the lesson."
                    }
                ],
                "summary": "To address conceptual misunderstandings, use collaborative learning, concrete representations, and formative assessment."
            }
        else:
            return {
                "strategies": [
                    {
                        "title": "Differentiated Instruction",
                        "description": "Provide multiple ways for students to access content, process information, and demonstrate learning."
                    },
                    {
                        "title": "Formative Assessment",
                        "description": "Use quick checks for understanding throughout the lesson to adjust your teaching in real-time."
                    },
                    {
                        "title": "Student Engagement",
                        "description": "Incorporate movement, choice, or technology to increase student participation and motivation."
                    }
                ],
                "summary": "Here are versatile strategies that can be adapted to various classroom situations and student needs."
            }
    
    try:
        print("🤖 Calling Groq LLM API...")
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert educational coach helping teachers with classroom challenges. Provide specific, actionable strategies in JSON format with 'strategies' (array of objects with 'title' and 'description') and 'summary' keys."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1024,
            response_format={"type": "json_object"}
        )
        
        # Parse JSON response
        content = response.choices[0].message.content
        print(f"✅ Received response from Groq LLM")
        
        try:
            parsed = json.loads(content)
            # Validate response structure
            if 'strategies' not in parsed:
                parsed['strategies'] = []
            if 'summary' not in parsed:
                parsed['summary'] = "Here are some strategies that might help:"
            return parsed
        except json.JSONDecodeError:
            print("⚠️  Could not parse JSON response, using fallback")
            return {
                "strategies": [
                    {
                        "title": "Response Format Issue",
                        "description": "The AI response couldn't be parsed properly. Please try rephrasing your question."
                    }
                ],
                "summary": content[:200] + "..." if len(content) > 200 else content
            }
        
    except Exception as e:
        print(f"❌ Error calling Groq API: {e}")
        # Return fallback response
        return {
            "strategies": [
                {
                    "title": "Technical Issue",
                    "description": "There was a problem connecting to the AI service. Please try again or use one of our pre-loaded strategies."
                }
            ],
            "summary": "I'm having trouble accessing the AI assistance right now. Here's a general strategy you can use."
        }

# Translation helper function - SAME AS OLD VERSION
def translate_text(text: str, target_lang: str, source_lang: str = 'auto') -> str:
    """Translate text to target language with better error handling."""
    try:
        if target_lang == 'en' or not text:
            return text
        
        # Handle language code variations for Telugu and other Indian languages
        lang_mapping = {
            'zh-cn': 'zh-CN',
            'zh': 'zh-CN',
            'te': 'te',  # Telugu
            'hi': 'hi',  # Hindi
            'ta': 'ta',  # Tamil
            'kn': 'kn',  # Kannada
            'ml': 'ml',  # Malayalam
            'bn': 'bn',  # Bengali
            'gu': 'gu',  # Gujarati
            'mr': 'mr',  # Marathi
            'pa': 'pa',  # Punjabi
            'ur': 'ur',  # Urdu
        }
        
        source_lang_adj = lang_mapping.get(source_lang, source_lang)
        target_lang_adj = lang_mapping.get(target_lang, target_lang)
        
        # For Indian languages, use 'auto' detection
        if target_lang_adj in ['te', 'hi', 'ta', 'kn', 'ml', 'bn', 'gu', 'mr', 'pa', 'ur']:
            source_lang_adj = 'auto'
        
        print(f"🌍 Translating {len(text)} chars: '{text[:50]}...'")
        print(f"   From: {source_lang_adj} → To: {target_lang_adj}")
        
        # Try translation
        translated = GoogleTranslator(source=source_lang_adj, target=target_lang_adj).translate(text)
        
        # Check if translation actually happened
        if translated == text:
            print(f"⚠️  Translation may have failed - same text returned")
            # Try with simpler approach
            try:
                # Split into sentences and translate individually
                import re
                sentences = re.split(r'(?<=[.!?])\s+', text)
                translated_sentences = []
                
                for sentence in sentences:
                    if len(sentence.strip()) > 0:
                        try:
                            sent_translated = GoogleTranslator(
                                source=source_lang_adj, 
                                target=target_lang_adj
                            ).translate(sentence)
                            translated_sentences.append(sent_translated)
                        except:
                            translated_sentences.append(sentence)
                
                translated = ' '.join(translated_sentences)
            except:
                pass
        
        print(f"🌍 Result: '{translated[:50]}...'")
        return translated
        
    except Exception as e:
        print(f"❌ Translation error: {e}")
        import traceback
        traceback.print_exc()
        # Return original text but log the error
        return text

def translate_cached(text: str, target_lang: str, source_lang: str = 'auto') -> str:
    if not text or target_lang == 'en':
        return text

    key = (hash(text), source_lang, target_lang)

    if key in translation_cache:
        return translation_cache[key]

    translated = translate_text(text, target_lang, source_lang)
    translation_cache[key] = translated
    return translated
    
# Text-to-speech function
def text_to_speech(text: str, lang: str) -> str:
    """Convert text to speech and return base64 encoded audio."""
    try:
        # Extended language code mapping for gTTS
        lang_map = {
            'en': 'en',
            'es': 'es',
            'fr': 'fr',
            'de': 'de',
            'hi': 'hi',
            'bn': 'bn',
            'ta': 'ta',
            'te': 'te',
            'ml': 'ml',
            'kn': 'kn',
            'gu': 'gu',
            'mr': 'mr',
            'pa': 'pa',
            'ur': 'ur',
            'ja': 'ja',
            'ko': 'ko',
            'zh-cn': 'zh-cn',
            'zh': 'zh-cn',
            'pt': 'pt',
            'ru': 'ru',
            'ar': 'ar',
            'it': 'it',
            'nl': 'nl'
        }
        
        tts_lang = lang_map.get(lang.lower(), 'en')
        
        print(f"🎤 TTS: Converting {len(text)} chars in {lang} (gTTS: {tts_lang})")
        
        # Split long text into chunks
        max_chars = 200
        if len(text) > max_chars:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= max_chars:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
        else:
            chunks = [text]
        
        print(f"🎤 Splitting into {len(chunks)} chunks")
        
        # Generate audio for each chunk
        audio_chunks = []
        for i, chunk in enumerate(chunks):
            try:
                tts = gTTS(text=chunk, lang=tts_lang, slow=False)
                
                # Use BytesIO to avoid file system operations
                audio_bytes = io.BytesIO()
                tts.write_to_fp(audio_bytes)
                audio_bytes.seek(0)
                audio_chunks.append(audio_bytes.read())
                
                print(f"🎤 Generated chunk {i+1}/{len(chunks)}")
                
            except Exception as chunk_error:
                print(f"⚠️ Error generating chunk {i+1}: {chunk_error}")
                # Try with English as fallback
                try:
                    tts = gTTS(text=chunk, lang='en', slow=False)
                    audio_bytes = io.BytesIO()
                    tts.write_to_fp(audio_bytes)
                    audio_bytes.seek(0)
                    audio_chunks.append(audio_bytes.read())
                    print(f"🎤 Used English fallback for chunk {i+1}")
                except:
                    continue
        
        if not audio_chunks:
            return ""
        
        # Combine all audio chunks
        combined_audio = b"".join(audio_chunks)
        
        # Encode to base64
        audio_base64 = base64.b64encode(combined_audio).decode('utf-8')
        
        print(f"✅ TTS successful: {len(audio_base64)} bytes encoded")
        return audio_base64
        
    except Exception as e:
        print(f"❌ Text-to-speech error: {e}")
        import traceback
        traceback.print_exc()
        return ""

# Routes
@app.route('/api/teachers', methods=['GET'])
def get_teachers():
    teachers = list(teachers_collection.find().sort("created_at", DESCENDING))
    return jsonify(serialize_docs(teachers))

@app.route('/api/classrooms', methods=['GET'])
def get_classrooms():
    classrooms = list(classrooms_collection.find().sort("created_at", DESCENDING))
    return jsonify(serialize_docs(classrooms))

@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    strategies = list(strategies_collection.find().sort("created_at", DESCENDING))
    return jsonify(serialize_docs(strategies))

@app.route('/api/interactions', methods=['GET'])
def get_interactions():
    interactions = list(interactions_collection.find().sort("created_at", DESCENDING))
    return jsonify(serialize_docs(interactions))

@app.route('/api/interactions', methods=['POST'])
def create_interaction():
    try:
        data = request.json
        
        interaction = {
            "teacher_id": data.get('teacher_id', ''),
            "classroom_id": data.get('classroom_id', ''),
            "problem_type": data.get('problem_type', 'other'),
            "query": data.get('query', ''),
            "advice": data.get('advice', []),
            "feedback": data.get('feedback', 'pending'),
            "escalated": data.get('escalated', False),
            "created_at": datetime.now(timezone.utc)
        }
        
        result = interactions_collection.insert_one(interaction)
        interaction['_id'] = str(result.inserted_id)
        
        print(f"✅ Created interaction: {interaction['_id']}")
        return jsonify(serialize_doc(interaction)), 201
        
    except Exception as e:
        print(f"❌ Error creating interaction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/interactions/<interaction_id>', methods=['PUT'])
def update_interaction(interaction_id):
    try:
        data = request.json
        
        update_data = {}
        if 'feedback' in data:
            update_data['feedback'] = data['feedback']
        
        result = interactions_collection.update_one(
            {"_id": ObjectId(interaction_id)},
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            updated = interactions_collection.find_one({"_id": ObjectId(interaction_id)})
            print(f"✅ Updated interaction: {interaction_id}")
            return jsonify(serialize_doc(updated))
        else:
            return jsonify({"error": "Interaction not found"}), 404
            
    except Exception as e:
        print(f"❌ Error updating interaction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/escalations', methods=['POST'])
def create_escalation():
    try:
        data = request.json
        
        escalation = {
            "interaction_id": data.get('interaction_id'),
            "teacher_id": data.get('teacher_id', ''),
            "reason": data.get('reason', ''),
            "level": data.get('level', 'mentor'),
            "status": data.get('status', 'pending'),
            "created_at": datetime.now(timezone.utc)
        }
        
        result = escalations_collection.insert_one(escalation)
        escalation['_id'] = str(result.inserted_id)
        
        print(f"✅ Created escalation: {escalation['_id']}")
        return jsonify(serialize_doc(escalation)), 201
        
    except Exception as e:
        print(f"❌ Error creating escalation: {e}")
        return jsonify({"error": str(e)}), 500

# Main coaching advice endpoint - FIXED LIKE OLD VERSION
@app.route('/api/coaching/advice', methods=['POST'])
def get_coaching_advice():
    try:
        data = request.json
        query = data.get('query', '')
        user_lang = data.get('user_lang', 'en')
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        print(f"📝 Processing query: {query[:50]}...")
        print(f"🌍 User language: {user_lang}")
        
        # Translate query to English if needed
        original_query = query
        translated_query = query
        
        if user_lang != 'en':
            try:
                translated_query = translate_cached(query, 'en', user_lang)
                print(f"🌍 Translated query to English: {translated_query[:50]}...")
            except Exception as e:
                print(f"⚠️ Query translation failed: {e}")
                translated_query = query
        
        # Get all necessary data
        teacher = teachers_collection.find_one()
        classroom = classrooms_collection.find_one({"teacher_id": str(teacher['_id'])}) if teacher else None
        strategies = list(strategies_collection.find())
        interactions = list(interactions_collection.find().sort("created_at", DESCENDING))
        
        # Convert to dictionaries
        teacher_dict = serialize_doc(teacher) if teacher else {}
        classroom_dict = serialize_doc(classroom) if classroom else {}
        strategies_dict = serialize_docs(strategies)
        interactions_dict = serialize_docs(interactions)
        
        print(f"📊 Context: Teacher={teacher_dict.get('name', 'None')}, Strategies={len(strategies_dict)}, Interactions={len(interactions_dict)}")
        
        # Use coaching orchestrator
        problem_type = classify_problem(translated_query)
        print(f"🔍 Problem classified as: {problem_type}")
        
        selection_result = select_strategies(
            strategies_dict,
            problem_type,
            interactions_dict,
            teacher_dict.get('confidence_level', 'medium')
        )
        
        print(f"🎯 Selected {len(selection_result['strategies'])} strategies")
        
        prompt = build_prompt(
            translated_query,
            problem_type,
            selection_result['strategies'],
            teacher_dict,
            classroom_dict,
            interactions_dict
        )
        
        # Get LLM response from Groq
        llm_response = call_groq_llm(prompt)
        
        # Ensure response has required structure
        if 'strategies' not in llm_response:
            llm_response['strategies'] = []
        if 'summary' not in llm_response:
            llm_response['summary'] = "Here are some strategies that might help:"
        
        print(f"✅ Generated {len(llm_response['strategies'])} strategies from LLM")
        
        # Create response - USING OLD VERSION TRANSLATION LOGIC
        response_data = {
            'problem_type': problem_type,
            'selected_strategy_ids': [s['_id'] for s in selection_result['strategies']],
            'should_escalate': selection_result['should_escalate'],
            'original_query': original_query,
            'translated_query': translated_query,
            'user_lang': user_lang,
        }
        
        # Translate response to user's language if needed - OLD VERSION LOGIC
        translated_response = {
            'strategies': llm_response['strategies'],
            'summary': llm_response['summary'],
            'original_summary': llm_response['summary']
        }
        
        if user_lang != 'en':
            try:
                # Translate summary
                translated_summary = translate_cached(llm_response['summary'], user_lang, 'en')
                
                # Translate strategies
                translated_strategies = []
                for strategy in llm_response['strategies']:
                    translated_title = translate_cached(strategy['title'], user_lang, 'en')
                    translated_desc = translate_cached(strategy['description'], user_lang, 'en')
                    translated_strategies.append({
                        'title': translated_title,
                        'description': translated_desc,
                        'original_title': strategy['title'],
                        'original_description': strategy['description']
                    })
                
                translated_response = {
                    'strategies': translated_strategies,
                    'summary': translated_summary,
                    'original_summary': llm_response['summary']
                }
                print(f"🌍 Translated response to {user_lang}")
                
            except Exception as e:
                print(f"⚠️ Failed to translate response: {e}")
                # Keep English response if translation fails
                for strategy in llm_response['strategies']:
                    strategy['original_title'] = strategy['title']
                    strategy['original_description'] = strategy['description']
                translated_response['original_summary'] = llm_response['summary']
        else:
            # Store original even if English
            for strategy in llm_response['strategies']:
                strategy['original_title'] = strategy['title']
                strategy['original_description'] = strategy['description']
            translated_response['original_summary'] = llm_response['summary']
        
        # Add translated response to main response
        response_data.update({
            'strategies': translated_response['strategies'],
            'summary': translated_response['summary'],
            'original_summary': translated_response.get('original_summary', '')
        })
        
        # Debug output
        print(f"📦 Response prepared:")
        print(f"   - Strategies count: {len(response_data['strategies'])}")
        print(f"   - Summary length: {len(response_data['summary'])}")
        print(f"   - User language: {user_lang}")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error in coaching advice endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Failed to generate coaching advice',
            'details': str(e)
        }), 500

# New endpoints for translation and voice
@app.route('/api/translate', methods=['POST'])
def translate_text_endpoint():
    try:
        data = request.json
        text = data.get('text', '')
        target_lang = data.get('target_lang', 'en')
        source_lang = data.get('source_lang', 'auto')
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        translated = translate_cached(text, target_lang, source_lang)
        
        return jsonify({
            "original": text,
            "translated": translated,
            "target_lang": target_lang,
            "source_lang": source_lang
        })
        
    except Exception as e:
        print(f"❌ Error in translation endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech_endpoint():
    try:
        data = request.json
        text = data.get('text', '')
        lang = data.get('lang', 'en')
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        # Generate speech
        audio_base64 = text_to_speech(text, lang)
        
        if not audio_base64:
            return jsonify({"error": "Failed to generate speech"}), 500
        
        return jsonify({
            "audio": audio_base64,
            "text": text,
            "lang": lang,
            "text_length": len(text)
        })
        
    except Exception as e:
        print(f"❌ Error in text-to-speech endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/speech-to-text', methods=['POST'])
def speech_to_text_endpoint():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "Audio file is required"}), 400
        
        audio_file = request.files['audio']
        lang = request.form.get('lang', 'en-US')
        
        # Save temporary audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            
            # Initialize recognizer
            recognizer = sr.Recognizer()
            
            # Convert audio to text
            with sr.AudioFile(temp_file.name) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language=lang)
            
            # Clean up
            os.unlink(temp_file.name)
        
        return jsonify({
            "text": text,
            "lang": lang
        })
        
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio"}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Speech recognition error: {e}"}), 500
    except Exception as e:
        print(f"❌ Error in speech-to-text endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/languages', methods=['GET'])
def get_supported_languages():
    """Get list of supported languages."""
    languages = {
        'en': {'name': 'English', 'code': 'EN', 'voice': 'en-US', 'tts': 'en'},
        'es': {'name': 'Spanish', 'code': 'ES', 'voice': 'es-ES', 'tts': 'es'},
        'fr': {'name': 'French', 'code': 'FR', 'voice': 'fr-FR', 'tts': 'fr'},
        'de': {'name': 'German', 'code': 'DE', 'voice': 'de-DE', 'tts': 'de'},
        'hi': {'name': 'Hindi', 'code': 'HI', 'voice': 'hi-IN', 'tts': 'hi'},
        'bn': {'name': 'Bengali', 'code': 'BN', 'voice': 'bn-IN', 'tts': 'bn'},
        'ta': {'name': 'Tamil', 'code': 'TA', 'voice': 'ta-IN', 'tts': 'ta'},
        'te': {'name': 'Telugu', 'code': 'TE', 'voice': 'te-IN', 'tts': 'te'},
        'ml': {'name': 'Malayalam', 'code': 'ML', 'voice': 'ml-IN', 'tts': 'ml'},
        'kn': {'name': 'Kannada', 'code': 'KN', 'voice': 'kn-IN', 'tts': 'kn'},
        'gu': {'name': 'Gujarati', 'code': 'GU', 'voice': 'gu-IN', 'tts': 'gu'},
        'mr': {'name': 'Marathi', 'code': 'MR', 'voice': 'mr-IN', 'tts': 'mr'},
        'pa': {'name': 'Punjabi', 'code': 'PA', 'voice': 'pa-IN', 'tts': 'pa'},
        'ur': {'name': 'Urdu', 'code': 'UR', 'voice': 'ur-PK', 'tts': 'ur'},
        'ja': {'name': 'Japanese', 'code': 'JA', 'voice': 'ja-JP', 'tts': 'ja'},
        'ko': {'name': 'Korean', 'code': 'KO', 'voice': 'ko-KR', 'tts': 'ko'},
        'zh-cn': {'name': 'Chinese', 'code': 'ZH', 'voice': 'zh-CN', 'tts': 'zh-cn'},
        'pt': {'name': 'Portuguese', 'code': 'PT', 'voice': 'pt-BR', 'tts': 'pt'},
        'ru': {'name': 'Russian', 'code': 'RU', 'voice': 'ru-RU', 'tts': 'ru'},
        'ar': {'name': 'Arabic', 'code': 'AR', 'voice': 'ar-SA', 'tts': 'ar'},
        'it': {'name': 'Italian', 'code': 'IT', 'voice': 'it-IT', 'tts': 'it'},
        'nl': {'name': 'Dutch', 'code': 'NL', 'voice': 'nl-NL', 'tts': 'nl'}
    }
    
    return jsonify(languages)

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        # Test MongoDB connection
        client.admin.command('ping')
        mongodb_status = 'connected'
    except:
        mongodb_status = 'disconnected'
    
    return jsonify({
        'status': 'healthy',
        'mongodb': mongodb_status,
        'groq': 'configured' if GROQ_API_KEY else 'not_configured',
        'groq_client': 'available' if groq_client else 'unavailable',
        'timestamp': datetime.now(timezone.utc).isoformat()
    })

# Root endpoint
@app.route('/')
def index():
    return jsonify({
        'name': 'Flash Coach API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'GET /api/health': 'Health check',
            'GET /api/teachers': 'Get all teachers',
            'GET /api/strategies': 'Get all strategies',
            'GET /api/interactions': 'Get all interactions',
            'POST /api/coaching/advice': 'Get coaching advice',
            'POST /api/translate': 'Translate text',
            'POST /api/text-to-speech': 'Text to speech',
            'POST /api/speech-to-text': 'Speech to text',
            'GET /api/languages': 'Get supported languages'
        }
    })

if __name__ == '__main__':
    print("🚀 Starting Flash Coach API Server...")
    print(f"📁 Database: {DATABASE_NAME}")
    print(f"🔑 Groq API: {'✅ Configured' if GROQ_API_KEY else '⚠️  Not configured'}")
    print(f"🤖 Groq Client: {'✅ Available' if groq_client else '⚠️  Using mock responses'}")
    
    init_db()
    
    print(f"🌐 Server running on http://localhost:{PORT}")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=PORT)
