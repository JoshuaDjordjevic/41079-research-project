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
fp_model_tomato_indices = os.path.join(fp_models, 'tomato.json')

models = ModelManager(
    fp_model_potato,
    fp_model_strawberry,
    fp_model_tomato,
    fp_model_tomato_indices)

@app.route('/')
def index():
    return render_template('index.html', result="result textbox", image_url=None)

@app.route('/analyze', methods=['POST'])
def analyze():
    image = request.files.get('image')
    result = "No image uploaded."
    image_data = None

    if image:
        # Read and encode the image
        img_bytes = image.read()
        encoded = base64.b64encode(img_bytes).decode('utf-8')
        image_data = f"data:{image.mimetype};base64,{encoded}"

        # Save image to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
            temp_img.write(img_bytes)
            temp_img_path = temp_img.name

        # Predict using potato model
        try:
            prediction = models.predict_potato(temp_img_path)
            result = formatting.format_prediction(prediction)
        except Exception as e:
            result = f"Error during prediction: {e}"
        finally:
            os.remove(temp_img_path)

    return render_template('index.html', result=result, image_data=image_data)

if __name__ == '__main__':
    app.run("0.0.0.0", debug=True)