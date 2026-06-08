import os
import json
import logging
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from markitdown import MarkItDown

app = Flask(__name__)
# Explicitly allow secure cross-origin resource sharing across local loops
CORS(app) 

# 🧭 Structured Chat Route configuration for advanced context token control
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
md_converter = MarkItDown()

# Configure uniform low-overhead logging matrix
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@app.route('/generate-answer', methods=['POST'])
def generate_answer():
    try:
        # 🛡️ Guardrail 1: Extract data silently to keep Flask from blowing up on weird request syntax
        data = request.get_json(silent=True)
        
        if not data:
            logging.warn("⚠️ Received empty or malformed request payload metadata structure.")
            return jsonify({
                "answer": "INSUFFICIENT_CONTEXT",
                "status": "MALFORMED_JSON_REQUEST"
            }), 200 # Always return 200 so NestJS handles it elegantly without breaking the pipeline!

        user_question = data.get("question", "")
        retrieved_context = data.get("context", "")

        # Coerce to string safely to protect against type errors during string normalization
        user_question = str(user_question).strip() if user_question is not None else ""
        retrieved_context = str(retrieved_context).strip() if retrieved_context is not None else ""

        if not user_question:
            return jsonify({
                "answer": "⚠️ System Constraint: Question input field cannot be blank.",
                "status": "MISSING_INPUT_QUESTION"
            }), 200

        if not retrieved_context:
            retrieved_context = "No relevant source document sections were extracted for this query match."

        # Structured context boundaries to keep Llama 3 constrained to reality
        system_instruction = (
            "You are a strict data extraction engine. Answer the user's question using ONLY the verbatim "
            "facts and phrases provided in the Context below. Do NOT hallucinate, do NOT use outside knowledge, "
            "and do NOT use introductory phrases like 'Based on the text'. If the answer is missing, reply with 'INSUFFICIENT_CONTEXT'."
        )

        # 🧠 Structured Chat Matrix payload setup to isolate user inputs from systemic instructions
        payload = {
            "model": "llama3",
            "messages": [
                {
                    "role": "system",
                    "content": system_instruction
                },
                {
                    "role": "user",
                    "content": f"--- CONTEXT START ---\n{retrieved_context}\n--- CONTEXT END ---\n\nQuestion: {user_question}"
                }
            ],
            "options": {
                "temperature": 0.0  # Completely deterministic execution matrix
            },
            "stream": False
        }

        # ⏱️ Guardrail 2: Extended 90-second connection gate window to handle long model attention stalls
        try:
            logging.info(f"📡 Forwarding context tokens directly to local Ollama core layer...")
            response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=90)
            response.raise_for_status()
            ollama_data = response.json()
            
            # ✨ FIXED: Hyper-defensive nested dictionary extraction fallback path
            message_obj = ollama_data.get("message")
            if isinstance(message_obj, dict):
                ai_response = message_obj.get("content", "").strip()
            else:
                ai_response = ""
                
        except Exception as ollama_err:
            logging.error(f"❌ Downstream Ollama Engine Failure: {str(ollama_err)}")
            return jsonify({
                "answer": "INSUFFICIENT_CONTEXT",
                "status": "OLLAMA_NODE_OFFLINE"
            }), 200

        # Standardize empty evaluations seamlessly to activate the NestJS intercept rules cleanly
        if not ai_response or ai_response.upper() == "INSUFFICIENT_CONTEXT.":
            ai_response = "INSUFFICIENT_CONTEXT"

        return jsonify({
            "answer": ai_response,
            "status": "COMPUTE_SUCCESS"
        }), 200

    except Exception as e:
        # 🛡️ Ultimate Guardrail: Complete capture catch-all to prevent a hard 500 server drop
        logging.critical(f"💥 Critical Exception caught inside Python Framework: {str(e)}")
        return jsonify({
            "answer": "INSUFFICIENT_CONTEXT",
            "status": "PYTHON_INTERNAL_FALLBACK"
        }), 200


@app.route('/api/extract-pdf', methods=['POST'])
def extract_pdf():
    file_path = None
    filename = "UNKNOWN"
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file stream detected in payload metadata"}), 400
            
        uploaded_file = request.files['file']
        filename = uploaded_file.filename if uploaded_file.filename else "UNKNOWN"
        
        if filename == '':
            return jsonify({"error": "Empty file sequence header passed"}), 400

        file_path = os.path.join(UPLOAD_FOLDER, filename)
        uploaded_file.save(file_path)

        logging.info(f"📄 Processing unstructured document ingestion: {filename}")
        conversion_result = md_converter.convert(file_path)
        
        return jsonify({
            "status": "SUCCESS",
            "filename": filename,
            "markdown": conversion_result.text_content
        })

    except Exception as e:
        logging.critical(f"❌ MarkItDown Ingestion Error for file {filename}: {str(e)}")
        return jsonify({"error": f"Extraction framework failure: {str(e)}"}), 500
    finally:
        # Guarantee removal of local residual binary file streams
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass


if __name__ == '__main__':
    logging.info("🤖 System Bootstrap Done. Listening on address: http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)