import glob
import os.path

import cv2
import mediapipe as mp
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import LSTM, Dense, Dropout, Flatten, TimeDistributed
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
    include_top=False,
    input_shape=(299, 299, 3)
)

x = base_model.output
x = Flatten()(x)
predictions = Dense(2, activation='softmax')(x)

# We'll extract features at the final pool layer.
inception_model = Model(
    inputs=base_model.input,
    outputs=predictions
)
sequence = []
image_name = 'SOLTERO.mp4'
cap = cv2.VideoCapture(image_name)
currentframe = 0
with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.4) as hands:
    nb_frames = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Total Frames: {total_frames}')
    for frame_num in range(total_frames):
        # print(frame_num)
        ret, frame = cap.read()
        # cv2.imshow('ventana1', frame)
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
        # print(f'Results: \n {results.multi_hand_landmarks}')
        class_name = ""
        probability = 0.0
        if results.multi_hand_landmarks:
            # print(f'Hay resultados')
            nb_frames += 1
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1,
                                                                 circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1,
                                                                 circle_radius=1),
                                          )
            if nb_frames <= 150:
                x = np.expand_dims(image, axis=0)
                x = preprocess_input(x)
                features = inception_model.predict(x, verbose=0)
                sequence.append(features[0])
                print(f"clases: {classes}")
                predicted_class = np.argmax(features)
                print(f"predicted_class: {predicted_class}")
                probability = features[0][predicted_class]
                print(f"probability: {probability:.2f}")
                class_name = classes[predicted_class]
                print(f"class_name: {class_name}")

            else:
                break

            print(nb_frames)

        text = f"{class_name}: {probability:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # cv2.imshow('Solucion', frame)


    if nb_frames < 150:
        for i in range(nb_frames + 1, 151, 1):
            x = np.expand_dims(image, axis=0)
            x = preprocess_input(x)
            features = inception_model.predict(x, verbose=0)
            sequence.append(features[0])

sequence = np.array([sequence])
prediction = model.predict(sequence)
maxid = np.argmax(prediction)

print(image_name, ' ------- ', classes[maxid])
