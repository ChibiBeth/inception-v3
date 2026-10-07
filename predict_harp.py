import argparse
import glob
import os.path

import numpy as np
from keras.models import load_model

from extract_features_harp import build_extractor, canvases_to_rgb, extract_features
from handtrack import process_video
from lspy_common import load_meta

parser = argparse.ArgumentParser(description='Predice la seña de un video con la LSTM entrenada.')
parser.add_argument('video', help='Ruta del video (por ejemplo ROJO.mp4)')
parser.add_argument('--model', default='lstm_senha_model')
parser.add_argument('--top', type=int, default=3, help='Cantidad de señas más probables a mostrar')
args = parser.parse_args()

model = load_model(args.model)
meta = load_meta(args.model)
if meta is not None:
    classes = meta['classes']
    data_type = meta['data_type']
    seq_length = meta['seq_length']
else:
    # Modelos entrenados antes de guardar el .json: se deducen de data/train
    print('[aviso] No se encontró %s.json; se usan las carpetas de data/train y features2048.' % args.model)
    classes = sorted(os.path.basename(c) for c in glob.glob(os.path.join('data', 'train', '*')))
    data_type, seq_length = 'features2048', 150

# Same steps as handtrack.py + extract_features_harp.py (+ retrain_inception.py)
info = process_video(args.video, seq_length)
if info['total_frames'] == 0:
    raise SystemExit('No se pudo leer el video %s' % args.video)
print('Fotogramas del video: %d; con manos en la secuencia: %d de %d'
      % (info['total_frames'], info['with_hand'], seq_length))

if data_type == 'landmarks':
    sequence = info['landmarks']
elif data_type in ('features2048', 'probs'):
    sequence = extract_features(build_extractor(), canvases_to_rgb(info['canvases']))
    if data_type == 'probs':
        head = load_model(os.path.join('data', 'inception_head.keras'))
        sequence = head.predict(sequence, verbose=0)
        frame_votes = sequence.argmax(1)
        print('Seña más probable por fotograma según Inception V3 (reentrenada): %s'
              % classes[np.bincount(frame_votes, minlength=len(classes)).argmax()])
else:
    raise SystemExit("La representación '%s' no se puede reproducir en la predicción." % data_type)

prediction = model.predict(sequence[np.newaxis], verbose=0)[0]
for i in np.argsort(prediction)[::-1][:args.top]:
    print('%-12s %.3f' % (classes[i], prediction[i]))

print(args.video, ' ------- ', classes[int(np.argmax(prediction))])
