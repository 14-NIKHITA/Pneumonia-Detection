import os
from flask import Flask, request, render_template, redirect
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Initialize the Flask application
app = Flask(__name__)

# Set the upload folder path
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')  # Make sure this points to your upload form

# Route for handling image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    img_file = request.files['file']
    
    if img_file:
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
        img_file.save(img_path)

        # Preprocess the image for prediction
        img = image.load_img(img_path, target_size=(150, 150))  
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize the image (if needed)

        # Load your trained model
        model = load_model('pneumonia_detection_model.h5')

        # Print the input shape to confirm the expected image dimensions
        print("Model input shape:", model.input_shape)

        # Predict the class using the loaded model
        prediction = model.predict(img_array)

        # Since it's binary classification, the output will be a single value
        # Interpreting the prediction result
        prediction_value = prediction[0][0]
        if prediction_value >= 0.5:
            result = "Pneumonia"
            confidence = prediction_value * 100
        else:
            result = "Normal"
            confidence = (1 - prediction_value) * 100

        # Render result.html with the prediction result and image
        return render_template('result.html', 
                               image_path='/static/uploads/' + img_file.filename, 
                               prediction=result,
                               confidence=f"{confidence:.2f}%")

    return "Invalid file format!"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
