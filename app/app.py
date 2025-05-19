import base64
import os
import tempfile
from flask import Flask, request, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename
from scripts import ModelManager, formatting

app = Flask(__name__)

fp_root = os.path.dirname(__file__)
fp_models = os.path.join(fp_root, 'models')
fp_model_potato = os.path.join(fp_models, 'potato.pth')
fp_model_strawberry = os.path.join(fp_models, 'strawberry.h5')
fp_model_tomato = os.path.join(fp_models, 'tomato.h5')

models = ModelManager(
    fp_model_potato,
    fp_model_strawberry,
    fp_model_tomato)

@app.route('/')
def index():
    return render_template('index.html', result="result textbox", image_url=None)

@app.route('/analyze', methods=['POST'])
def analyze():
    image = request.files.get('image')
    model_type = request.form.get('model', 'potato')  # Default to potato
    result = "No image uploaded."
    image_data = None

    if image:
        # Save the uploaded image to a temporary path
        filename = secure_filename(image.filename)
        temp_path = os.path.join(fp_root, 'temp_' + filename)
        image.save(temp_path)

        # Read and encode the image for preview
        with open(temp_path, 'rb') as img_file:
            encoded = base64.b64encode(img_file.read()).decode('utf-8')
            image_data = f"data:{image.mimetype};base64,{encoded}"

        # Predict using selected model
        try:
            if model_type == 'potato':
                prediction = formatting.format_prediction(models.predict_potato(temp_path))
            elif model_type == 'strawberry':
                prediction = formatting.format_prediction(models.predict_strawberry(temp_path))
            elif model_type == 'tomato':
                prediction = formatting.format_prediction(models.predict_tomato(temp_path))
            else:
                prediction = "Unknown model selected."

            result = prediction
        except Exception as e:
            result = f"Error: {e}"

        # Clean up temporary image
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return render_template('index.html', result=result, image_data=image_data)


if __name__ == '__main__':
    app.run("0.0.0.0", debug=True)