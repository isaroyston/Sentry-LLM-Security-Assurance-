from flask import Flask, render_template, request, jsonify
import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.chatbot.withdrawal_chatbot import WithdrawalChatbot
from src.vector_store.vector_store import VectorStore
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Initialize vector store once when starting the app
vs = VectorStore(
    persist_directory="../vectordb",
    collection_name="sgbank_withdrawal_policy"
)
print(f"DEBUG: Initialized VectorStore at ../vectordb. Collection count: {vs.get_collection_count()}")
bot = WithdrawalChatbot(vector_store=vs)

@app.route('/')
def index():
    # Renders the HTML template
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Call the actual chatbot logic
    try:
        response = bot.chat(user_message)
    except Exception as e:
        response = f"Error: {str(e)}"
    
    return jsonify({"response": response})

if __name__ == '__main__':
    # Running Flask in debug mode for development
    app.run(debug=True, port=3000)
