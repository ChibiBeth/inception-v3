import argparse
import csv
import glob
import os

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from lspy_common import frame_pattern, sequence_file

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
CANVAS_SIZE = 299
DRAWING_SPEC = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)


def get_video_parts(video_path, vid_dir):
    # funciona con rutas relativas, absolutas o de varios niveles (H12)
    train_or_test, classname, filename = os.path.relpath(video_path, vid_dir).split(os.sep)[-3:]
    # nombre de archivo de video sin extension (splitext respeta nombres con puntos)
    filename_no_ext = os.path.splitext(filename)[0]
    return train_or_test, classname, filename_no_ext, filename


def new_hands():
    return mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.4)


def detect_hands(frame, hands):
    """Detecta las manos sobre el cuadro completo, sin deformarlo (H8).

    El cuadro se espeja (igual que en la versión original, para que entrenamiento
    y predicción coincidan) y se completa con bordes negros hasta hacerlo
    cuadrado; así las coordenadas normalizadas de MediaPipe se pueden dibujar
    en el lienzo de 299 x 299 sin cambiar la proporción de la mano.
    """
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    square = cv2.copyMakeBorder(frame, top, s - h - top, left, s - w - left,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    image = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    return hands.process(image)


def draw_hands(results, size=CANVAS_SIZE):
    """Dibujo blanco de los puntos y conexiones sobre fondo negro (BGR, uint8)."""
    img = np.zeros((size, size, 3), np.uint8)
    if results is not None and results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS, DRAWING_SPEC, DRAWING_SPEC)
    return img


def landmarks_vector(results):
    """126 valores por fotograma: 2 manos x 21 puntos x (x, y, z); ceros si falta una mano (H9).

    La muñeca (punto 0) se guarda en coordenadas del cuadro y los otros 20 puntos
    relativos a la muñeca: la forma de la mano queda independiente de su posición,
    pero se conserva dónde está la mano respecto del cuerpo.
    """
    out = np.zeros((2, 21, 3), np.float32)
    if results is not None and results.multi_hand_landmarks:
        free = [0, 1]
        for hand, info in zip(results.multi_hand_landmarks, results.multi_handedness):
            k = 0 if info.classification[0].label == 'Left' else 1
            if k not in free:
                if not free:
                    break
                k = free[0]
            free.remove(k)
            pts = np.array([[p.x, p.y, p.z] for p in hand.landmark], np.float32)
            pts[1:] -= pts[0]
            out[k] = pts
    return out.ravel()


def process_video(video_path, n_frames=150):
    """Procesa un video completo y lo remuestrea por tiempo a `n_frames` (H7).

    Se ejecuta MediaPipe sobre todos los cuadros (el seguimiento necesita cuadros
    consecutivos) y luego se eligen `n_frames` cuadros equiespaciados entre el
    primero y el último. Los cuadros sin manos no se descartan: quedan en negro,
    así la secuencia conserva la escala temporal y videos de 30 y 60 fps son
    comparables. Devuelve los dibujos (BGR), las coordenadas (n_frames, 126) y
    datos del video.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    all_results = []
    with new_hands() as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = detect_hands(frame, hands)
            all_results.append(results if results.multi_hand_landmarks else None)
    cap.release()

    total = len(all_results)
    if total == 0:
        selected = [None] * n_frames
    else:
        idx = np.linspace(0, total - 1, num=n_frames).round().astype(int)
        selected = [all_results[i] for i in idx]

    return {
        'canvases': [draw_hands(r) for r in selected],
        'landmarks': np.stack([landmarks_vector(r) for r in selected]),
        'with_hand': sum(r is not None for r in selected),
        'total_frames': total,
        'fps': round(fps, 2),
    }


def hands_extraction(vid_dir, out_dir, n_frames=150, folders=('train', 'test')):
    data_file = []
    for folder in ('sequences', 'checkpoints', 'logs'):
        os.makedirs(os.path.join(out_dir, folder), exist_ok=True)

    # se crea una lista de todos los videos de cada clase (palabra) en train y test
    videos = []
    for folder in folders:
        for vid_class in sorted(glob.glob(os.path.join(vid_dir, folder, '*'))):
            videos += [f for f in sorted(glob.glob(os.path.join(vid_class, '*')))
                       if f.lower().endswith(VIDEO_EXTENSIONS)]

    for file_name in tqdm(videos, desc='Detectando manos'):
        train_or_test, classname, filename_no_ext, _ = get_video_parts(file_name, vid_dir)
        class_dir = os.path.join(out_dir, train_or_test, classname)
        os.makedirs(class_dir, exist_ok=True)

        info = process_video(file_name, n_frames)
        if info['total_frames'] == 0:
            tqdm.write(f'[aviso] No se pudo leer {file_name}; se omite.')
            continue

        # se borran los fotogramas y secuencias de una ejecución anterior de este video,
        # que ya no corresponden a los nuevos dibujos
        stale = glob.glob(os.path.join(class_dir, frame_pattern(filename_no_ext)))
        seq_dir = os.path.join(out_dir, 'sequences', train_or_test)
        os.makedirs(seq_dir, exist_ok=True)
        stale += glob.glob(os.path.join(seq_dir, glob.escape(filename_no_ext) + '-[0-9]*-*.npy'))
        for f in stale:
            os.remove(f)

        # se escribe cada fotograma en el directorio de salida que le corresponda a su clase
        for i, img in enumerate(info['canvases'], start=1):
            cv2.imwrite(os.path.join(class_dir, '{}-{}.jpg'.format(filename_no_ext, str(i).rjust(4, '0'))), img)
        np.save(sequence_file(out_dir, train_or_test, filename_no_ext, n_frames, 'landmarks'), info['landmarks'])

        # se guardan los datos relevantes de cada video procesado
        # (las columnas 5 a 7 son informativas: fotogramas con mano, fotogramas del video y fps)
        data_file.append([train_or_test, classname, filename_no_ext, n_frames,
                          info['with_hand'], info['total_frames'], info['fps']])

    # se escriben los datos relevantes de los videos procesados en el archivo data_file.csv
    with open(os.path.join(out_dir, 'data_file.csv'), 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)


def main():
    parser = argparse.ArgumentParser(
        description='Detecta las manos con MediaPipe y genera los dibujos y coordenadas por video.')
    parser.add_argument('-i', '--ifile', required=True, help='Carpeta con train/<Clase>/*.mp4 y test/<Clase>/*.mp4')
    parser.add_argument('-o', '--ofile', required=True, help='Carpeta de salida (por ejemplo data)')
    parser.add_argument('-n', '--frames', type=int, default=150,
                        help='Fotogramas por video tras remuestrear por tiempo (debe coincidir con --seq-length)')
    args = parser.parse_args()
    hands_extraction(args.ifile, args.ofile, args.frames)


if __name__ == '__main__':
    main()
