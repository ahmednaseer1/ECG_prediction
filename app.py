import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('ecg_model.h5')

# Prediction labels with details
index = [
    ('Left Bundle Branch Block', 'A condition where there is a delay or blockage along the pathway that electrical impulses travel to make your heart beat.'),
    ('Normal', 'The ECG is normal. No abnormalities detected.'),
    ('Premature Atrial Contraction', 'A type of irregular heartbeat that occurs when the atria contract earlier than usual.'),
    ('Premature Ventricular Contractions', 'Extra heartbeats that begin in the ventricles and disrupt the regular heart rhythm.'),
    ('Right Bundle Branch Block', 'A condition where there is a delay or blockage along the pathway that electrical impulses travel to make your heart beat, specifically on the right side.'),
    ('Ventricular Fibrillation', 'A life-threatening heart rhythm problem that occurs when the heart beats with rapid, erratic electrical impulses.')
]

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return "No file part", 400

    f = request.files['file']
    basepath = os.path.dirname(__file__)
    filepath = os.path.join(basepath, "uploads", f.filename)
    f.save(filepath)

    img = image.load_img(filepath, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)
    y_pred = np.argmax(pred)

    result, details = index[y_pred]

    # Include an HTML line break between the result and the details
    return f"{result}<br>{details}"

if __name__ == "__main__":
    app.run(debug=False)
