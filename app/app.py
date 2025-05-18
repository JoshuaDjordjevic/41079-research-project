import base64
import os
from flask import Flask, request, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

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

        # Placeholder logic
        result = "This plant appears healthy. (Placeholder result)"

    return render_template('index.html', result=result, image_data=image_data)

if __name__ == '__main__':
    app.run("0.0.0.0", debug=True)
