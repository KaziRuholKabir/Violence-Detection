import os
from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory
from werkzeug.utils import secure_filename
import cv2
from tqdm import tqdm
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
#app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.avi', '.mp4', '.mpg']
app.config['UPLOAD_PATH'] = 'uploads'

@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('index.html', files=files)

@app.route('/pred', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)

    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))

    cap = cv2.VideoCapture('./uploads/'+filename)
    success = True

    data = []
    count = 0
    while success:
        success, image = cap.read()
        if not success:
            break

        img = cv2.resize(image, (100, 100))
        data.append(np.array(img))
        count += 1

        if count == 10:
            break
    
    X = np.array(data).reshape(1, 10, 100, 100, 3)
    model = load_model('./Violence_detection-CNN-BiLSTM.h5')
    Y = model.predict(X)[0][0]
    print(Y)

    
    if Y > 0.5:
        return render_template('index.html', pred='The video is VIOLENT', vid= filename)
    else:
        return render_template('index.html', pred='The video is NOT VIOLENT', vid= filename)



@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

if __name__ == '__main__':
   app.run(debug = True)