from flask import Flask, render_template, request
import os
from yolo import detect_objects

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_PATH = "static/output.jpg"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']

        # Save uploaded file
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        # ✅ SAVE ORIGINAL IMAGE (IMPORTANT FIX)
        file.save("static/input.jpg")

        # Run detection
        detect_objects(
            path,
            OUTPUT_PATH,
            "yolo.cfg",
            "yolo.weights",
            "yolo.txt"
        )

        return render_template('index.html', output=True)

    return render_template('index.html', output=False)


if __name__ == '__main__':
    app.run(debug=True)