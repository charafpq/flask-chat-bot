import os
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from rag_pipeline import answer_question
from datetime import datetime
import traceback

load_dotenv()
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat.db'

db = SQLAlchemy(app)
class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(20))
    content = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
with app.app_context():
    db.create_all()
    
@app.route('/', methods=['GET'])
def home():   
    all_messages = Message.query.order_by(Message.timestamp).all()
    return render_template('index.html', messages=all_messages)

@app.route('/send', methods=['POST'])
def send():
    # Get the message from JSON or form data
    data = request.get_json() or {}
    # Get the user message from JSON or form data, and strip whitespace
    user_message = data.get('message', '').strip()
    # If the user message is empty, redirect to home
    if not user_message:
        return jsonify(error='Empty message'), 400

    assistant_response = answer_question(user_message)
    # Save user message
    db.session.add(Message(role="user", content=user_message))
    db.session.add(Message(role="assistant", content=assistant_response))
    db.session.commit()
    return jsonify({'assistant_response': assistant_response})

if __name__ == '__main__':
    app.run(debug=True)