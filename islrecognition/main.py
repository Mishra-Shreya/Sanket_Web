
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

# # Actions that we try to detect
# alpha_actions = np.array(['-','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
# words_actions = np.array(['-', 'hello', 'good', 'morning'])

# # # Thirty videos worth of data
# # no_sequences = 45
# # # Videos are going to be 10 frames in length
# # sequence_length = 10

# alpha_model = load_model('islrecognition\models\\alphabets_83606.h5')
# words_model = load_model('islrecognition\models\\words.h5')

# alpha_shape = 10
# words_shape = 15

# actions = words_actions
# shape = words_shape
# model = words_model

camera = cv2.VideoCapture(0)

def recognition(actions, shape, model):  # generate frame by frame from camera
    
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5
    count = 0
    prev = '-'

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
                sequence = sequence[-shape:]
                
                if len(sequence) == shape:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))
                    # prev = actions[np.argmax(res)]

                    if prev == ' '.join(actions[np.argmax(res)]): 
                        count += 1
                    else:
                        count = 0
                        prev = ' '.join(actions[np.argmax(res)])
                    
                    if res[np.argmax(res)] > threshold : 
                        sentence = actions[np.argmax(res)]
                        # print(''.join(actions[np.argmax(res)]))
            
                cv2.rectangle(image, (0,440), (640, 480), (0, 140, 255), -1)
                cv2.putText(image, 'Prediction : ', (3,470), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, ' '.join(sentence), (200,470), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Show to screen
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        

@main.route('/video_feed/<choice>', methods=['GET', 'POST'])
def video_feed(choice):
    # Actions that we try to detect
    alpha_actions = np.array(['-','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
    words_actions = np.array(['-', 'hello', 'good', 'morning'])

    alpha_model = load_model('islrecognition\models\\alphabets_83606.h5')
    words_model = load_model('islrecognition\models\\words.h5')

    alpha_shape = 10
    words_shape = 15
   
    if choice == 'words':
        actions = words_actions
        shape = words_shape
        model = words_model
    else:
        actions = alpha_actions
        shape = alpha_shape
        model = alpha_model

    #Video streaming route. Put this in the src attribute of an img tag
    return Response(recognition(actions, shape, model), mimetype='multipart/x-mixed-replace; boundary=frame')

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/islrecognition', methods=['GET', 'POST'])
def islrecognition():

    if request.method == 'POST':    
        if request.form['choice'] == 'words':
            choice = 'words'
        else:
            choice = 'alphabets'
    else:
        choice = 'words'
    
    return render_template('islrecognition.html', choice=choice)

@main.route('/speechtoisl', methods=['GET', 'POST'])
def speechtoisl():
    
    return render_template('speechtoisl.html')



