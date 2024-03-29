from flask import Flask,render_template,Response,request
from keras.models import load_model
import base64
app=Flask(__name__)
import io
from PIL import Image
import numpy as np
import pickle
import os
model=pickle.load(open('mil.pkl','rb'))
UPLOAD_FOLDER = r'C:\Users\devad\OneDrive\Desktop\MP\static\images'  # Specify the folder where you want to save the uploaded images
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def preprocess_image(img):
    # img = img.resize(img, (300, 300),resample=Image.BILINEAR)  # Resize image to match model input size
    img_resized = img.resize((300, 300), resample=Image.BILINEAR)

    # img_a=np.asarray(img_resized)
    img_a = np.array(img_resized) / 255.0 
    img_b = np.expand_dims(img_a, axis=0)  # Add batch dimension
    return img_b
@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/about')
def about_deficiencies():
    return render_template('about.html')

@app.route('/docs')
def about_us():
    return render_template('docs.html')

@app.route('/queries')
def queries():
    return render_template('queries.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index1.html', message='No file part')

    file = request.files['file']
    file_path=file.filename
    if file.filename == '':
        return render_template('index1.html', message='No selected file')

    if file:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        img=preprocess_image(img) 
        prediction=model.predict(img)
        print(prediction)
        predicted_class='Real' if prediction[0][0]>prediction[0][1] else 'Fake'
        print(predicted_class)
        # return predicted_class
        print(file_path)
        # filename = file.filename
        # file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # file.save(file_path)
        return render_template('index1.html', predicted_class=predicted_class)     

if __name__ == "__main__":
    app.run(debug=True)
# if __name__=="__main__":
#     app.run(debug=True)