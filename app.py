from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the model
model = load_model('model/lung_cancer_model.h5')

# Set up upload folder
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            filename = file.filename

            # Preprocess the image
            img = image.load_img(filepath, target_size=(64, 64), color_mode="grayscale")
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0  # Normalize to [0,1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Make prediction
            pred = model.predict(img_array)[0][0]
            print(f"Raw prediction score: {pred}")  # Debug output
            prediction = (f"Lung Cancer Detected ({pred*100:.2f}%)" if pred > 0.5 
                         else f"No Lung Cancer ({(1-pred)*100:.2f}%)")

    return render_template('index.html', prediction=prediction, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)