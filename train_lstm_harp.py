import argparse
import os
import time

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.layers import Dense, Dropout, Input, LSTM
from keras.models import Sequential
from keras.optimizers import Adam

from extract_features_harp import DataSet
from lspy_common import DATA_TYPES, grouped_train_val_split, save_meta, set_seed

import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

parser = argparse.ArgumentParser(description='Entrena la LSTM sobre las secuencias de data/sequences.')
parser.add_argument('--data-type', default='features2048', choices=sorted(DATA_TYPES),
                    help='Representación de cada fotograma (ver lspy_common.DATA_TYPES)')
parser.add_argument('--arch', default='ligera', choices=['ligera', 'original'],
                    help="'ligera': 2 LSTM (128, 64), recomendada para pocos datos; 'original': 3 LSTM (2048, 256, 128)")
parser.add_argument('--lr', type=float, default=1e-4, help='Tasa de aprendizaje de Adam (la versión original usaba 1e-5)')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--val-frac', type=float, default=0.2,
                    help='Fracción de videos originales de train reservada para validación')
parser.add_argument('--seq-length', type=int, default=150)
parser.add_argument('--class-limit', type=int, default=10)
parser.add_argument('--seed', type=int, default=42, help='Semilla de inicialización y orden de entrenamiento')
parser.add_argument('--split-seed', type=int, default=42,
                    help='Semilla de la separación train/validación (fija, para comparar semillas con la misma validación)')
parser.add_argument('--clases', nargs='+', default=None, help='Entrenar solo con estas señas (por defecto, todas)')
parser.add_argument('--sin-checkpoints', action='store_true',
                    help='No guardar checkpoints por época (el modelo final ya tiene los mejores pesos)')
parser.add_argument('--output', default='lstm_senha_model')
args = parser.parse_args()

set_seed(args.seed)
name = 'lstm-%s-%s' % (args.data_type, args.arch)
if args.output != 'lstm_senha_model':
    name += '-' + os.path.basename(args.output.rstrip('/'))

checkpointer = ModelCheckpoint(
    filepath=os.path.join('data', 'checkpoints', name + '.{epoch:03d}-{val_loss:.3f}.hdf5'),
    verbose=1,
    save_best_only=True)

# Helper: TensorBoard
tb = TensorBoard(log_dir=os.path.join('data', 'logs', name))

# Helper: Stop when we stop learning, and keep the best weights (not the last epoch's).
early_stopper = EarlyStopping(patience=args.patience, restore_best_weights=True)

# Helper: Save results.
os.makedirs(os.path.join('data', 'logs'), exist_ok=True)
os.makedirs(os.path.join('data', 'checkpoints'), exist_ok=True)
timestamp = time.time()
csv_logger = CSVLogger(os.path.join('data', 'logs', name + '-' + 'training-' + \
                                    str(timestamp) + '.log'))

# Get the data and process it.
data = DataSet(
    seq_length=args.seq_length,
    class_limit=args.class_limit,
    classes=args.clases
)
train_rows, test_rows = data.split_train_test()

# Validation comes from train, grouping each original video with its augmented copies,
# so the test set is used only once, at the end.
tr_idx, va_idx = grouped_train_val_split([r[2] for r in train_rows], [r[1] for r in train_rows],
                                         args.val_frac, args.split_seed)
X, y = data.load_rows([train_rows[i] for i in tr_idx], args.data_type)
X_val, y_val = data.load_rows([train_rows[i] for i in va_idx], args.data_type)
X_test, y_test = data.load_rows(test_rows, args.data_type)
print('Secuencias: entrenamiento %d, validación %d, prueba %d; forma de cada una %s'
      % (len(X), len(X_val), len(X_test), X.shape[1:]))

model = Sequential()
model.add(Input(shape=X.shape[1:]))
if args.arch == 'original':
    model.add(LSTM(2048, return_sequences=True, dropout=0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.5))
else:
    model.add(LSTM(128, return_sequences=True, dropout=0.3))
    model.add(LSTM(64, dropout=0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
model.add(Dense(len(data.classes), activation='softmax'))
optimizer = Adam(learning_rate=args.lr)
model.compile(loss='categorical_crossentropy', optimizer=optimizer,
              metrics=['accuracy', 'top_k_categorical_accuracy'])
model.summary()

model.fit(
    X,
    y,
    batch_size=args.batch_size,
    validation_data=(X_val, y_val),
    verbose=2,
    callbacks=[tb, early_stopper, csv_logger] + ([] if args.sin_checkpoints else [checkpointer]),
    epochs=args.epochs)

# The test set is evaluated only once, with the weights chosen on validation.
loss, acc, top5 = model.evaluate(X_test, y_test, verbose=0)
print('Prueba: exactitud %.3f (%d de %d), top-5 %.3f, pérdida %.3f'
      % (acc, round(acc * len(X_test)), len(X_test), top5, loss))

model.save(args.output)
save_meta(args.output, {
    'classes': data.classes,
    'data_type': args.data_type,
    'seq_length': args.seq_length,
    'arch': args.arch,
    'lr': args.lr,
    'seed': args.seed,
    'split_seed': args.split_seed,
    'log': name,
    'videos_validacion': sorted({train_rows[i][2] for i in va_idx}),
    'exactitud_prueba': acc,
})
print('Modelo guardado en %s (clases y configuración en %s.json)' % (args.output, args.output))
