import os
from flask import Flask, render_template, request, redirect, flash, send_from_directory, url_for
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from PIL import Image, ImageOps, ExifTags

# Setup Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = 'supersecretkey'
model = YOLO("models/fixxmodel.pt")  

def preprocess_image(image_path):
    img = Image.open(image_path)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation)
            if orientation_value == 3:
                img = img.rotate(180, expand=True)
            elif orientation_value == 6:
                img = img.rotate(270, expand=True)
            elif orientation_value == 8:
                img = img.rotate(90, expand=True)
    except Exception:
        pass

    img = ImageOps.exif_transpose(img) 
    img = img.resize((640, 640), Image.BILINEAR)
    img_cv = np.array(img.convert('L'))
    img_eq = cv2.equalizeHist(img_cv)
    img_rgb = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2RGB)

    return img_rgb

def predict_and_draw(img_path, output_path):
    img = preprocess_image(img_path)
    results = model(img)[0]     
    result_img = img.copy()

    predicted_class = "None"
    confidence = 0.0

    # Warna label
    label_colors = {
        '3': (255, 0, 0),
        '4': (0, 0, 255),
        '5': (0, 255, 0)
    }

    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            label = model.names[cls_id]

            if score > confidence:
                predicted_class = label
                confidence = score

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = label_colors.get(label, (255, 255, 0))
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result_img, f'{label}: {score:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(output_path, result_img)
    return predicted_class, confidence, output_path

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
            predicted_class, confidence, result_image_path = predict_and_draw(filepath, result_path)
            message = f"Predicted Class: PAI {predicted_class} with confidence {confidence:.2f}"
            flash(message)

            return render_template('result.html',
                                   result_image=os.path.basename(result_image_path),
                                   message=message)
        else:
            flash('Invalid file type. Please upload an image.')
            return redirect(request.url)

    return redirect(url_for('detection'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
