from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, Input
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from collections import deque
from keras.preprocessing import image as Img
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from tqdm import tqdm
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import extract_features_harp
import sys
import time
import os.path
import csv
import numpy as np
import random
import glob
import sys
import operator
import threading
import os 
import cv2


checkpointer = ModelCheckpoint(
    filepath=os.path.join('data', 'checkpoints','lstm-features' + '.{epoch:03d}-{val_loss:.3f}.hdf5'),
    verbose=1,
    save_best_only=True)

# Helper: TensorBoard
tb = TensorBoard(log_dir=os.path.join('data', 'logs', 'lstm'))

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=10)

# Helper: Save results.
timestamp = time.time()
csv_logger = CSVLogger(os.path.join('data', 'logs', 'lstm' + '-' + 'training-' + \
    str(timestamp) + '.log'))

# Get the data and process it.
data = extract_features_harp.DataSet(
    seq_length=40,
    class_limit=70
)
#listt=[]
#listt2=[]
X, y = data.get_all_sequences_in_memory('train', 'features')
X_test, y_test = data.get_all_sequences_in_memory('test', 'features')

model = Sequential()
model.add(LSTM(2048, return_sequences=False,input_shape=(40,2048),dropout=0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(data.classes), activation='softmax'))
optimizer = Adam(lr=1e-5, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                    metrics=['accuracy','top_k_categorical_accuracy'])
print(model.summary())

model.fit(
    X,
    y,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[tb, early_stopper, csv_logger,checkpointer],
    epochs=100)

model.save("lstm_senha_model")