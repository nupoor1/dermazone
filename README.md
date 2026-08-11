# dermazone

A Flask web app that classifies a user's skin type (oily, dry, or normal) from a photo using a custom-trained CNN, then recommends skincare products for that skin type.

## What it does

dermazone lets a user upload a photo of their face, runs it through a convolutional neural network trained on a labeled skin-type image dataset, and returns a prediction (oily / dry / normal). Logged-in users get their prediction saved to a history on their dashboard, and the app pulls matching products from a small catalog, ranked by category and price, so the result is immediately actionable rather than just a label.

## Key technical decisions

- **Small custom CNN instead of transfer learning.** `model1.py` trains a from-scratch CNN (two Conv2D+BatchNorm/MaxPooling blocks, `GlobalAveragePooling2D`, a 128-unit dense layer with dropout) on 150×150 images rather than fine-tuning a pretrained network. Global average pooling instead of `Flatten` keeps the parameter count down, which matters given the training set is only ~2,750 images across 3 classes.
- **Model loaded once at process start.** `main.py` loads `skin_types.keras` as a module-level global when Flask boots, so `/predict` requests reuse the already-loaded model instead of paying deserialization cost per request.
- **Product ranking via an explicit scoring heuristic.** `score_product()` in `main.py` assigns points for category (moisturizer > cleanser > other) and price tier, then the top 3 scored products for the predicted skin type are returned.

## Tech stack

- **Backend:** Flask, Flask-SQLAlchemy (SQLite)
- **ML:** TensorFlow/Keras (CNN), Pillow + NumPy for image preprocessing
- **Auth:** Werkzeug password hashing, Flask sessions, Flask-WTF CSRF protection
- **Frontend:** Jinja2, Tailwind CSS + daisyUI (via CDN), JavaScript

## Limitations

- **Move configuration to environment variables.** `SECRET_KEY` and the database URI are currently set directly in `main.py`; reading them from the environment would keep sessions valid across restarts and make the app deployable without code changes.
- **Add a proper role/permissions system.** Admin access is currently a hardcoded username check (`session['username'] == 'admin'`); a role field on `User` would be more robust and support more than one admin.
- **Harden form input validation.** Routes like `/admin/add-product` currently assume well-formed input (e.g. a parseable float for price); adding explicit validation would make the app more resilient to bad input.
- **Add automated tests** for the auth, prediction, and product routes.
- **Add a seed script for the product catalog** so it isn't limited to manual entry through the admin form.

## Setup and run

```bash
pip install -r requirements.txt
python main.py
```

This starts the Flask dev server (`app.run(debug=True)`) on `http://127.0.0.1:5000`. On first run, `db.create_all()` creates `instance/users.db` automatically if it doesn't already exist.

Notes:
- `skin_types.keras` (the trained model) is already committed to the repo root, so you don't need to retrain to run the app.
- The product catalog starts empty. To add products, register an account with the username **`admin`** (the `/admin/add-product` route is gated on that exact username) and use the form there — there's no seed script or bulk import.
- To retrain the model instead of using the committed one, download the [Kaggle Oily/Dry/Normal skin types dataset](https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset) and extract it into `train/`, `test/`, and `valid/` folders in the project root (these aren't tracked in the repo), then run `python model1.py`, which writes `skin_types.keras`.
