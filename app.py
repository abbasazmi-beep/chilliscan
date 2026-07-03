from flask import Flask, render_template, request, jsonify
import os
import uuid
import json
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# ── Load Models ───────────────────────────────────────────────
print("Loading models...")

with open('models/svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

mobilenet_model = tf.keras.models.load_model('models/mobilenet_model.keras')

with open('models/class_indices.json', 'r') as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

print("✅ Models loaded!")

# ── Helpers ───────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_for_svm(image_path):
    img = Image.open(image_path).convert('RGB').resize((64, 64))
    arr = np.array(img).flatten() / 255.0
    return arr.reshape(1, -1)

def preprocess_for_mobilenet(image_path):
    img = Image.open(image_path).convert('RGB').resize((96, 96))
    arr = np.array(img) / 255.0
    return arr.reshape(1, 96, 96, 3)

def get_color(label):
    return {
        'Ripe': 'ripe',
        'Semi-Ripe': 'semi-ripe',
        'Unripe': 'unripe',
        'Overripe': 'overripe'
    }.get(label, 'ripe')

# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    model_type = request.form.get('model', 'mobilenet')

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not supported. Use JPG or PNG."}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        if model_type == 'svm':
            X = preprocess_for_svm(filepath)
            pred_idx = svm_model.predict(X)[0]
            label = label_encoder.inverse_transform([pred_idx])[0]
            proba = svm_model.predict_proba(X)[0]
            confidence = round(float(np.max(proba)) * 100, 1)
            model_name = "SVM"
        else:
            X = preprocess_for_mobilenet(filepath)
            preds = mobilenet_model.predict(X, verbose=0)[0]
            pred_idx = int(np.argmax(preds))
            label = idx_to_class[pred_idx]
            confidence = round(float(np.max(preds)) * 100, 1)
            model_name = "MobileNetV2"

        return jsonify({
            "label": label,
            "confidence": confidence,
            "color": get_color(label),
            "model": model_name,
            "image_url": f"/static/uploads/{filename}"
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
