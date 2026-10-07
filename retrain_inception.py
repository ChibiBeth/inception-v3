"""Reentrenamiento de la última capa de Inception V3 (transfer learning).

Inception V3 (ImageNet) queda congelada y se entrena solo una capa softmax nueva
sobre sus 2048 características, con un fotograma como ejemplo y la seña del video
como etiqueta (como el clásico `retrain.py` de TensorFlow). Así Inception V3
cumple el rol que describe la tesis: extrae características y entrega, para cada
fotograma, la probabilidad de pertenencia a cada seña. Esas probabilidades se
guardan como `data/sequences/<partición>/<video>-<T>-probs.npy` para entrenar la LSTM con
`--data-type probs`.

Requiere haber ejecutado antes extract_features_harp.py (secuencias 'features2048').

Para evitar fugas de información:
  - la capa se entrena solo con los videos de entrenamiento de la LSTM (se excluye
    la misma validación agrupada por video original que usa train_lstm_harp.py, y
    nunca se usa test);
  - las probabilidades de los videos de entrenamiento se calculan "fuera de
    pliegue" (cada video recibe las de una capa que no lo vio), para que la LSTM
    no aprenda con probabilidades más confiadas que las que verá en validación,
    prueba o predicción.
"""
import argparse
import os
import random

import numpy as np
from keras import Sequential, regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam

from extract_features_harp import DataSet, build_extractor, canvases_to_rgb, extract_features
from lspy_common import grouped_train_val_split, original_stem, save_meta, set_seed

HEAD_PATH = os.path.join('data', 'inception_head.keras')


def build_head(n_features, n_classes, lr):
    model = Sequential([
        Input(shape=(n_features,)),
        Dropout(0.5),
        Dense(n_classes, activation='softmax', kernel_regularizer=regularizers.l2(1e-4)),
    ])
    model.compile(optimizer=Adam(lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def frames_and_labels(X, y, idx, hand_mask):
    """Aplana las secuencias de los videos `idx` a fotogramas, sin los fotogramas vacíos."""
    m = hand_mask[idx].ravel()
    return X[idx].reshape(-1, X.shape[-1])[m], np.repeat(y[idx], X.shape[1])[m]


def video_accuracy(probs, y, hand_mask):
    """Exactitud por video promediando las probabilidades de los fotogramas con mano."""
    hits = []
    for p, label, m in zip(probs, y, hand_mask):
        p = p[m] if m.any() else p
        hits.append(p.mean(0).argmax() == label)
    return float(np.mean(hits)) if hits else float('nan')


def main():
    parser = argparse.ArgumentParser(description='Reentrena la última capa de Inception V3 y genera las secuencias de probabilidades.')
    parser.add_argument('--seq-length', type=int, default=150)
    parser.add_argument('--class-limit', type=int, default=10)
    parser.add_argument('--val-frac', type=float, default=0.2, help='Debe coincidir con train_lstm_harp.py')
    parser.add_argument('--folds', type=int, default=5, help='Pliegues para las probabilidades fuera de pliegue')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    data = DataSet(seq_length=args.seq_length, class_limit=args.class_limit)
    rows = data.data
    X, y_hot = data.load_rows(rows, 'features2048')
    X = X.astype(np.float32)
    y = y_hot.argmax(1)
    K = len(data.classes)

    # Fotogramas sin manos (dibujo negro): no aportan a la capa nueva
    black = extract_features(build_extractor(), canvases_to_rgb([np.zeros((299, 299, 3), np.uint8)]))[0]
    hand_mask = np.abs(X - black).max(-1) > 1e-3

    train_idx = [i for i, r in enumerate(rows) if r[0] == 'train']
    test_idx = [i for i, r in enumerate(rows) if r[0] != 'train']
    tr, va = grouped_train_val_split([rows[i][2] for i in train_idx], [rows[i][1] for i in train_idx],
                                     args.val_frac, args.seed)
    tr_idx, va_idx = [train_idx[i] for i in tr], [train_idx[i] for i in va]
    print('Videos: entrenamiento %d, validación %d, prueba %d' % (len(tr_idx), len(va_idx), len(test_idx)))

    # 1. Capa final con todos los videos de entrenamiento; la validación decide cuándo parar
    Xf, yf = frames_and_labels(X, y, tr_idx, hand_mask)
    Xv, yv = frames_and_labels(X, y, va_idx, hand_mask)
    head = build_head(X.shape[-1], K, args.lr)
    stopper = EarlyStopping(patience=5, restore_best_weights=True)
    hist = head.fit(Xf, yf, validation_data=(Xv, yv) if len(yv) else None, batch_size=256,
                    epochs=args.epochs, callbacks=[stopper] if len(yv) else [], verbose=2)
    best_epochs = int(np.argmin(hist.history['val_loss'])) + 1 if len(yv) else args.epochs
    head.save(HEAD_PATH)

    probs = head.predict(X.reshape(-1, X.shape[-1]), batch_size=1024, verbose=0).reshape(len(X), X.shape[1], K)

    # 2. Probabilidades fuera de pliegue para los videos de entrenamiento
    groups = sorted({original_stem(rows[i][2]) for i in tr_idx})
    random.Random(args.seed).shuffle(groups)
    fold_of = {g: n % args.folds for n, g in enumerate(groups)}
    for k in range(args.folds):
        fit_idx = [i for i in tr_idx if fold_of[original_stem(rows[i][2])] != k]
        out_idx = [i for i in tr_idx if fold_of[original_stem(rows[i][2])] == k]
        if not out_idx:
            continue
        fold_head = build_head(X.shape[-1], K, args.lr)
        fold_head.fit(*frames_and_labels(X, y, fit_idx, hand_mask), batch_size=256, epochs=best_epochs, verbose=0)
        probs[out_idx] = fold_head.predict(X[out_idx].reshape(-1, X.shape[-1]), batch_size=1024,
                                           verbose=0).reshape(len(out_idx), X.shape[1], K)
        print('Pliegue %d/%d listo' % (k + 1, args.folds))

    for i, row in enumerate(rows):
        np.save(data.sequence_file(row, 'probs'), probs[i].astype(np.float32))

    # Resultado de Inception V3 sola (sin LSTM), útil como línea base en la tesis
    for name, idx in (('validación', va_idx), ('prueba', test_idx)):
        if idx:
            m = hand_mask[idx]
            frame_acc = float((probs[idx].argmax(-1)[m] == np.repeat(y[idx], X.shape[1])[m.ravel()]).mean())
            print('Inception V3 reentrenada, %s: exactitud por fotograma %.3f; por video (promedio) %.3f'
                  % (name, frame_acc, video_accuracy(probs[idx], y[idx], m)))

    save_meta(HEAD_PATH, {'classes': data.classes, 'seq_length': args.seq_length, 'seed': args.seed,
                          'val_frac': args.val_frac, 'epocas': best_epochs,
                          'videos_validacion': sorted({rows[i][2] for i in va_idx})})
    print('Capa guardada en %s; secuencias -probs.npy en %s' % (HEAD_PATH, data.sequence_path))


if __name__ == '__main__':
    main()
