import glob
import os.path
import csv
from builtins import set
import mediapipe as mp

import cv2
import keras.utils as Img
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model

seq_lenght = 40

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def rescale_list(input_list, size):
    assert len(input_list) >= size
    skip = len(input_list) // size
    output = [input_list[i] for i in range(0, len(input_list), skip)]
    return output[:size]


model = load_model("lstm_senha_model")
classes = glob.glob(os.path.join('data', 'train', '*'))
classes = [classes[i].split('/')[2] for i in range(len(classes))]
classes = sorted(classes)

base_model = InceptionV3(
    weights='imagenet',
    include_top=True
)
# We'll extract features at the final pool layer.
inception_model = Model(
    inputs=base_model.input,
    outputs=base_model.get_layer('avg_pool').output
)
sequence = []
image_name = 'marzo_test.avi'
cap = cv2.VideoCapture(image_name)
currentframe = 0
with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.4) as hands:
    nb_frames = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Total Frames: {total_frames}')
    for frame_num in range(total_frames):
        print(frame_num)
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if not ret:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break
        frame = cv2.resize(frame, (299, 299))

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # FONDO NEGRO
        color = (0, 0, 0)
        # IMAGEN DE 860x720 x3 canales
        image = np.full((299, 299, 3), color, np.uint8)

        # Detections

        # Rendering results
        print(f'Results: \n {results.multi_hand_landmarks}')

        if results.multi_hand_landmarks:
            print(f'Hay resultados')
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1,
                                                                 circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1,
                                                                 circle_radius=1),
                                          )
            img = Img.load_img(image, target_size=(299, 299))
            x = Img.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = inception_model.predict(x, verbose=0)
            sequence.append(features[0])

            sequence = np.array([sequence])
            prediction = model.predict(sequence)
            maxm = prediction[0][0]
            maxid = 0
            for i in range(len(prediction[0])):
                if (maxm < prediction[0][i]):
                    maxm = prediction[0][i]
                    maxid = i

            cv2.putText(image, f"Class: {classes[maxid]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2, cv2.LINE_AA)

            print(image_name, ' ------- ', classes[maxid])
            cv2.imshow('frame', image)
