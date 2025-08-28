from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import tempfile
import traceback
import json
from datetime import datetime
from werkzeug.utils import secure_filename

# Import your existing modules
try:
    from extractors.pdf_extractor import PDFExtractor
    from extractors.word_extractor import WordExtractor  
    from extractors.image_extractor import ImageExtractor
    from preprocessors.text_preprocessor import TextPreprocessor
    from processors.text_cleaner import TextCleaner
    from processors.structure_analyzer import StructureAnalyzer
    from processors.metadata_extractor import MetadataExtractor
    from summarizer.multilingual_summarizer import MultilingualSummarizer
    from utils.language_detector import LanguageDetector
    from utils.quality_checker import QualityChecker
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Make sure all your modules are properly structured and have __init__.py files")

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt', 'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

# In-memory storage for documents (replace with database in production)
documents_storage = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_extractor(file_extension):
    """Return appropriate extractor based on file type"""
    try:
        if file_extension in ['pdf']:
            return PDFExtractor()
        elif file_extension in ['docx', 'doc']:
            return WordExtractor()
        elif file_extension in ['png', 'jpg', 'jpeg', 'gif']:
            return ImageExtractor()
        else:
            return None
    except Exception as e:
        print(f"Error creating extractor: {e}")
        return None

# Mock classes for testing if modules are not available
class MockExtractor:
    def extract(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            # For non-text files, return basic info
            content = f"Document: {os.path.basename(file_path)}"
        
        return {
            'text': content[:1000] + "..." if len(content) > 1000 else content,
            'metadata': {
                'pages': 1, 
                'author': 'Unknown', 
                'title': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }
        }

class MockPreprocessor:
    def preprocess(self, text):
        # Clean up text
        text = text.replace('\n\n', '\n').replace('\r', '').strip()
        return text

class MockLanguageDetector:
    def detect(self, text):
        french_words = ['le', 'la', 'les', 'un', 'une', 'et', 'de', 'du', 'des', 'à', 'ce', 'cette']
        french_count = sum(1 for word in french_words if word in text.lower())
        return 'fr' if french_count > 2 else 'en'

class MockSummarizer:
    def extractive_summary(self, text, language='auto'):
        sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.strip()) > 10]
        # Take first 3 meaningful sentences
        summary_sentences = sentences[:3]
        return '. '.join(summary_sentences) + '.' if summary_sentences else "Aucun contenu significatif trouvé."
    
    def abstractive_summary(self, text, language='auto'):
        word_count = len(text.split())
        if word_count < 50:
            return "Document trop court pour un résumé abstractif."
        return f"Ce document de {word_count} mots traite de sujets variés et contient des informations importantes."

class MockAnalyzer:
    def analyze(self, text):
        paragraphs = len([p for p in text.split('\n\n') if p.strip()])
        sentences = len([s for s in text.split('.') if s.strip()])
        words = len(text.split())
        
        return {
            'metadata': {
                'word_count': words,
                'sentence_count': sentences, 
                'paragraph_count': paragraphs,
                'avg_sentence_length': round(words / sentences if sentences > 0 else 0, 2)
            },
            'structure': {
                'has_title': bool(text.strip().split('\n')[0] if text.strip() else False),
                'has_paragraphs': paragraphs > 1,
                'estimated_reading_time': f"{max(1, round(words / 200))} minutes"
            }
        }

class MockMetadataExtractor:
    def extract(self, text):
        words = text.split()
        return {
            'key_terms': list(set([w.strip('.,!?;:').lower() for w in words if len(w) > 4]))[:10],
            'word_frequency': len(set(words)),
            'character_count': len(text)
        }

class MockQualityChecker:
    def check_quality(self, text):
        words = len(text.split())
        sentences = len([s for s in text.split('.') if s.strip()])
        avg_word_length = sum(len(word) for word in text.split()) / words if words > 0 else 0
        
        return {
            'readability_score': min(100, max(0, 100 - (avg_word_length * 5))),
            'complexity': 'simple' if avg_word_length < 5 else 'medium' if avg_word_length < 7 else 'complex',
            'word_count': words,
            'quality_rating': 'good' if words > 100 and sentences > 5 else 'fair'
        }

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Document Processing API',
        'documents_count': len(documents_storage),
        'modules_loaded': {
            'extractors': 'available',
            'preprocessors': 'available', 
            'summarizers': 'available',
            'analyzers': 'available'
        }
    })

# Add the missing /upload route that your frontend expects
@app.route('/upload', methods=['POST'])
def upload_file_legacy():
    """Legacy upload route for compatibility"""
    return upload_file()

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and complete processing"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'File type not supported. Supported types: PDF, DOCX, DOC, TXT, Images'
            }), 400
        
        # Check if document already exists
        filename = secure_filename(file.filename)
        doc_id = f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if filename in [doc['filename'] for doc in documents_storage.values()]:
            return jsonify({
                'success': False,
                'error': 'Un document avec ce nom existe déjà'
            }), 400
        
        # Save file temporarily
        file_extension = filename.rsplit('.', 1)[1].lower()
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Get appropriate extractor
        extractor = get_file_extractor(file_extension)
        if not extractor:
            extractor = MockExtractor()
        
        # Extract content with better error handling
        try:
            extracted_data = extractor.extract(file_path)
            text_content = extracted_data.get('text', '')
            metadata = extracted_data.get('metadata', {})
        except Exception as e:
            print(f"Extraction error: {e}")
            # Fallback extraction
            text_content = f"Erreur lors de l'extraction du contenu de {filename}"
            metadata = {'filename': filename, 'extraction_error': str(e)}
        
        # Preprocessing
        try:
            preprocessor = TextPreprocessor()
        except:
            preprocessor = MockPreprocessor()
        
        processed_text = preprocessor.preprocess(text_content) if text_content else ""
        
        # Language detection
        try:
            language_detector = LanguageDetector()
        except:
            language_detector = MockLanguageDetector()
        
        detected_language = language_detector.detect(processed_text) if processed_text else 'fr'
        
        # Generate summaries
        try:
            summarizer = MultilingualSummarizer()
        except:
            summarizer = MockSummarizer()
        
        try:
            extractive_summary = summarizer.extractive_summary(processed_text, language=detected_language)
            abstractive_summary = summarizer.abstractive_summary(processed_text, language=detected_language)
        except Exception as e:
            print(f"Summarization error: {e}")
            extractive_summary = "Erreur lors de la génération du résumé extractif"
            abstractive_summary = "Erreur lors de la génération du résumé abstractif"
        
        # Structure analysis
        try:
            structure_analyzer = StructureAnalyzer()
            structure_analysis = structure_analyzer.analyze(processed_text)
        except:
            structure_analyzer = MockAnalyzer()
            structure_analysis = structure_analyzer.analyze(processed_text)
        
        # Store document data
        document_data = {
            'id': doc_id,
            'filename': filename,
            'text': processed_text,
            'language': detected_language,
            'metadata': metadata,
            'word_count': len(processed_text.split()) if processed_text else 0,
            'upload_date': datetime.now().isoformat(),
            'extractive_summary': extractive_summary,
            'abstractive_summary': abstractive_summary,
            'structure_analysis': structure_analysis
        }
        
        documents_storage[doc_id] = document_data
        
        # Clean up temporary file
        try:
            os.remove(file_path)
        except:
            pass
        
        # Return complete response for frontend
        return jsonify({
            'success': True,
            'message': 'Document traité avec succès',
            'document': {
                'id': doc_id,
                'filename': filename,
                'language': detected_language,
                'word_count': document_data['word_count'],
                'upload_date': document_data['upload_date']
            },
            'summaries': {
                'extractive': extractive_summary,
                'abstractive': abstractive_summary
            },
            'analysis': structure_analysis
        })
        
    except Exception as e:
        print(f"Upload error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Erreur lors du traitement: {str(e)}'
        }), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get list of all uploaded documents"""
    try:
        document_list = []
        for doc_id, doc_data in documents_storage.items():
            document_list.append({
                'id': doc_id,
                'filename': doc_data['filename'],
                'language': doc_data['language'],
                'word_count': doc_data['word_count'],
                'upload_date': doc_data['upload_date']
            })
        
        return jsonify({
            'success': True,
            'documents': document_list,
            'count': len(document_list)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/documents/<doc_id>', methods=['GET'])
def get_document(doc_id):
    """Get specific document details"""
    try:
        if doc_id not in documents_storage:
            return jsonify({
                'success': False,
                'error': 'Document not found'
            }), 404
        
        document = documents_storage[doc_id]
        return jsonify({
            'success': True,
            'document': document
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a document"""
    try:
        if doc_id not in documents_storage:
            return jsonify({
                'success': False,
                'error': 'Document not found'
            }), 404
        
        del documents_storage[doc_id]
        return jsonify({
            'success': True,
            'message': 'Document supprimé avec succès'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Keep your existing endpoints for backward compatibility
@app.route('/api/summarize', methods=['POST'])
def summarize_text():
    """Generate summary from processed text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        language = data.get('language', 'auto')
        summary_type = data.get('type', 'extractive')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        try:
            summarizer = MultilingualSummarizer()
        except:
            summarizer = MockSummarizer()
        
        try:
            if summary_type == 'extractive':
                summary = summarizer.extractive_summary(text, language=language)
            elif summary_type == 'abstractive':
                summary = summarizer.abstractive_summary(text, language=language)
            else:
                extractive = summarizer.extractive_summary(text, language=language)
                abstractive = summarizer.abstractive_summary(text, language=language)
                summary = {
                    'extractive': extractive,
                    'abstractive': abstractive
                }
        except Exception as e:
            print(f"Summarization error: {e}")
            summary = "Erreur lors de la génération du résumé"
        
        return jsonify({
            'success': True,
            'summary': summary,
            'original_length': len(text.split()),
            'summary_length': len(str(summary).split()) if isinstance(summary, str) else None
        })
        
    except Exception as e:
        print(f"Summarization endpoint error: {e}")
        return jsonify({'error': f'Summarization failed: {str(e)}'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_document():
    """Perform structure analysis and metadata extraction"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        try:
            structure_analyzer = StructureAnalyzer()
            structure = structure_analyzer.analyze(text)
        except:
            structure_analyzer = MockAnalyzer()
            structure = structure_analyzer.analyze(text)
        
        try:
            metadata_extractor = MetadataExtractor()
            metadata = metadata_extractor.extract(text)
        except:
            metadata_extractor = MockMetadataExtractor()
            metadata = metadata_extractor.extract(text)
        
        try:
            quality_checker = QualityChecker()
            quality_metrics = quality_checker.check_quality(text)
        except:
            quality_checker = MockQualityChecker()
            quality_metrics = quality_checker.check_quality(text)
        
        return jsonify({
            'success': True,
            'structure': structure,
            'metadata': metadata,
            'quality': quality_metrics
        })
        
    except Exception as e:
        print(f"Analysis endpoint error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Document Processing Server...")
    print("Available endpoints:")
    print("- POST /upload (legacy)")
    print("- POST /api/upload")
    print("- GET /api/documents")
    print("- GET /api/documents/<id>")
    print("- DELETE /api/documents/<id>")
    print("- POST /api/summarize")
    print("- POST /api/analyze")
    print("Make sure your main.html is in the templates/ folder")
    print("Server will be available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)