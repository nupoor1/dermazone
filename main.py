from flask import Flask, render_template, request, redirect, session, url_for, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import random
import tensorflow as tf
import numpy as np
import io
from PIL import Image
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from datetime import datetime
import secrets

app = Flask(__name__)
model = tf.keras.models.load_model('skin_types.keras')

# Configure SQLAlchemy
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    username = db.Column(db.String(25), unique = True, nullable = False)
    password_hash = db.Column(db.String(150), nullable = False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Dynamic welcome messages
messagelist = ["Welcome, ", "Hope you’re well, ", "Let’s get cooking, ", "Let’s learn about your skin, "]
displaymessage = random.choice(messagelist)

# Routes
@app.route('/')
def home():
    if "username" in session:
        return redirect(url_for('dashboard'))
    return render_template('homepage.html')

@app.route('/index')
def index():
    return render_template('index.html')

# Login
@app.route("/login", methods =["POST"])
def login():
    username = request.form['username']
    password = request.form['password']
    user = User.query.filter_by(username=username).first()

    if user and user.check_password(password):
        session['username'] = username
        return redirect(url_for('dashboard'))
    if not user:
        error = 'User does not exist!'
        return render_template('index.html', error=error)
    else:
        error = 'Incorrect username or password.'
        return render_template('index.html', error=error)

# Register
@app.route("/register", methods =["POST"])
def register():
    username = request.form['username']
    password = request.form['password']
    user = User.query.filter_by(username=username).first()

    if not username:
        error = 'Username cannot be empty!'
        return render_template('index.html', error=error)

    if user:
        error = 'User already exists!'
        return render_template('index.html', error=error)
    else:
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        session['username'] = username
        return redirect(url_for('dashboard'))


# Dashboard
@app.route('/dashboard')
def dashboard():
    if "username" in session:
        user = User.query.filter_by(username=session['username']).first()
        predictions = Prediction.query.filter_by(user_id=user.id).all()
        return render_template('dashboard.html', username=session['username'], predictions=predictions, welcome_message=displaymessage)
    return redirect(url_for('home'))


# Logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))


# CNN IMPLEMENTATIONS
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('predictions', lazy=True))
    prediction = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Prediction {self.prediction} on {self.timestamp}>'



from datetime import datetime

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((150, 150))
        img_array = np.array(img)

        img_array = img_array / 255.0

        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)

        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        class_names = ['oily', 'dry', 'normal']
        prediction = class_names[class_index]

        # Save prediction to database
        if 'username' in session:
            user = User.query.filter_by(username=session['username']).first()
            new_prediction = Prediction(user_id=user.id, prediction=prediction)
            db.session.add(new_prediction)
            db.session.commit()

        return jsonify({'prediction': prediction})

    return jsonify({'error': 'Invalid file'}), 400



if __name__ in "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)