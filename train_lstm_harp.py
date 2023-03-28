import os
import time

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.optimizers import Adam

import extract_features_harp

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

checkpointer = ModelCheckpoint(
    filepath=os.path.join('data', 'checkpoints', 'lstm-features' + '.{epoch:03d}-{val_loss:.3f}.hdf5'),
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
# listt=[]
# listt2=[]
X, y = data.get_all_sequences_in_memory('train', 'features')
X_test, y_test = data.get_all_sequences_in_memory('test', 'features')

model = Sequential()
model.add(LSTM(2048, return_sequences=False, input_shape=(40, 2048), dropout=0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(data.classes), activation='softmax'))
optimizer = Adam(learning_rate=1e-5, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=optimizer,
              metrics=['accuracy', 'top_k_categorical_accuracy'])
print(model.summary())

model.fit(
    X,
    y,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=3,
    callbacks=[tb, early_stopper, csv_logger, checkpointer],
    epochs=100)

print('final')
model.save("lstm_senha_model")
