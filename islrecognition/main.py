
from flask import Blueprint, render_template 
from flask import Flask, render_template, Response, request
import cv2
from .isl import mp_holistic, mediapipe_detection, draw_styled_landmarks, extract_keypoints
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

main = Blueprint('main', __name__)


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Actions that we try to detect
actions = np.array(['-','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
# Thirty videos worth of data
no_sequences = 45
# Videos are going to be 10 frames in length
sequence_length = 10

model = load_model('islrecognition\models\\alphabets_83606.h5')

def recognition():  # generate frame by frame from camera
    
    camera = cv2.VideoCapture(0)
    
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while camera.isOpened():
            # Capture frame-by-frame
            success, frame = camera.read()  # read the camera frame
            if not success:
                break
            else:
                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                # print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-10:]
                
                if len(sequence) == 10:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))
                    
                    if res[np.argmax(res)] > threshold: 
                        sentence = actions[np.argmax(res)]
            
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Show to screen
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        

@main.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/islrecognition', methods=['GET', 'POST'])
def islrecognition():
    
    return render_template('islrecognition.html')



