from flask import Flask,render_template,Response,request
app=Flask(__name__)
import cv2
import os
import numpy as np
import pickle
model=pickle.load(open('mil_webcam.pkl','rb'))
cam=cv2.VideoCapture(0)
# Define the boundary coordinates
boundary_width = 470
boundary_height = 180
boundary_center_x = int((640 - boundary_width) / 2)
boundary_center_y = int((480 - boundary_height) / 2)
top_left = (boundary_center_x, boundary_center_y)
bottom_right = (boundary_center_x + boundary_width, boundary_center_y + boundary_height)

saved_folder = r"C:\Users\devad\OneDrive\Desktop\Testing"
current_frame=0

def preprocess_image(img):
    img = cv2.resize(img, (300, 300))  # Resize image to match model input size
    img=np.asarray(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def generate():
    global current_frame
    while True:
        success,frame= cam.read()
        if not success:
            print("not working")
            break
        else:
            # frame =cv2.flip(frame,1)
            # Create a mask to disable the region outside the boundary
            mask = np.zeros_like(frame)
            cv2.rectangle(mask, top_left, bottom_right, (255, 255, 255), -1)
            disabled_region = cv2.bitwise_and(frame, mask)
            # Draw the boundary rectangle on the frame
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            # Show the disabled region within the original frame
            result_frame = cv2.addWeighted(disabled_region, 0.5, frame, 0.5, 0)
            # Crop the frame using the specified boundary
            cropped_frame = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            ret, buffer=cv2.imencode('.jpg',result_frame)
            frame_bytes=buffer.tobytes()

            # cv2.imwrite(filename,cropped_frame)
            # Capture the image when space key is pressed
            key = cv2.waitKey(1)
            if key == ord(' '):
                # filename=f"{saved_folder}/frame_{current_frame}.jpg"
                filename = os.path.join(saved_folder, f'frame_{current_frame}.jpg')
                cv2.imwrite(filename, cropped_frame)
                current_frame += 1

            yield(b'--frame\r\n'
                  b'content-type: image/jpeg\r\n\r\n'+frame_bytes+b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/close', methods=['POST'])
def close():
    # Release the webcam resources
    cam.release()
    return '', 204

@app.route('/capture',methods=['POST'])
def capture():
    global current_frame
    # Capture a frame when the space key is pressed
    success, frame = cam.read()
    if success:
        cropped_frame = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        
        # filename = os.path.join(saved_folder, f'frame_{request.form["frame_id"]}.jpg')
        filename = os.path.join(saved_folder, f'frame_{current_frame}.jpg')
        cv2.imwrite(filename, cropped_frame)
        current_frame += 1
        s=True

    saved_image=cv2.imread(filename)
    processed_image=preprocess_image(saved_image)
    # model1=model.load_weights("Final_model.weights.h5")
    prediction=model.predict(processed_image)
    predicted_class='Fake' if prediction[0][0]>prediction[0][1] else 'Real'
    print(predicted_class)
    print(current_frame)
    # return render_template('ans.html',)
    # if(s==True):
    #     return render_template('ans.html', predicted_class=predicted_class)
    return predicted_class

if __name__=="__main__":
    app.run(debug=True)