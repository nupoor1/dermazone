from flask import Flask, render_template, request, redirect, session, url_for, jsonify
from flask_wtf import CSRFProtect
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
csrf = CSRFProtect(app)

# Database Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    username = db.Column(db.String(25), unique = True, nullable = False)
    password_hash = db.Column(db.String(150), nullable = False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    brand = db.Column(db.String(100), nullable=True)
    skin_type = db.Column(db.String(50), nullable=False)  # oily, dry, normal
    category = db.Column(db.String(50), nullable=True)
    price = db.Column(db.Float, nullable=True)
    image_url = db.Column(db.String(200), nullable=True)

    def __repr__(self):
        return f'<Product {self.name} ({self.skin_type})>'

# Dynamic welcome messages
messagelist = ["Welcome, ", "Hope you’re well, ", "Let’s get cooking, ", "Let’s learn about your skin, "]
displaymessage = random.choice(messagelist)

def score_product(product):
    score = 0

    # Prefer moisturizers
    if product.category:
        if product.category.lower() == "moisturizer":
            score += 3
        elif product.category.lower() == "cleanser":
            score += 2
        else:
            score += 1

    # Prefer affordable products
    if product.price:
        if product.price < 15:
            score += 3
        elif product.price < 25:
            score += 2
        else:
            score += 1

    return score

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
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        class_names = ['dry', 'normal', 'oily']
        prediction = class_names[class_index]

        # Save prediction to database
        if 'username' in session:
            user = User.query.filter_by(username=session['username']).first()
            new_prediction = Prediction(user_id=user.id, prediction=prediction)
            db.session.add(new_prediction)
            db.session.commit()

        # Fetch recommended products for this skin type
        products = Product.query.filter_by(skin_type=prediction).all()
        ranked_products = sorted(
            products,
            key=lambda p: score_product(p),
            reverse=True
            )
        top_products = ranked_products[:3]

        product_list = [
            {
        'name': p.name,
        'brand': p.brand,
        'category': p.category,
        'price': p.price,
        'image_url': p.image_url
    } for p in top_products
]

        return jsonify({'prediction': prediction, 'products': product_list})

    return jsonify({'error': 'Invalid file'}), 400

@app.route('/admin/add-product', methods=['GET', 'POST'])
def add_product():
    if 'username' not in session:
        return redirect(url_for('home'))

    # simple protection: only allow your account
    if session['username'] != 'admin':
        return "Access denied", 403

    if request.method == 'POST':
        name = request.form['name']
        brand = request.form['brand']
        skin_type = request.form['skin_type']
        category = request.form['category']
        price = request.form['price']
        image_url = request.form['image_url']

        product = Product(
            name=name,
            brand=brand,
            skin_type=skin_type,
            category=category,
            price=float(price),
            image_url=image_url
        )

        db.session.add(product)
        db.session.commit()

        return redirect(url_for('add_product'))

    return render_template('add_product.html')


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)