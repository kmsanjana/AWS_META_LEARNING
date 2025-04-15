from flask import Flask, render_template, request
from inference import predict
from PIL import Image
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    label = None
    if request.method == 'POST':
        file = request.files['file']
        file.save('static/uploaded.jpg')  # Save image
        label = predict('static/uploaded.jpg')  # Run inference
    return render_template('index.html', label=label)

if __name__ == '__main__':
    app.run(debug=True)

