"""Aumento de datos de video: rotación, zoom y eliminación aleatoria de fotogramas.

Por cada video original genera `--max-clips` copias '<nombre>_<i>.mp4'. Cada copia:
  - rota todos sus cuadros un mismo ángulo aleatorio en [-max_rotacion, +max_rotacion];
  - aplica un mismo zoom aleatorio en [1 - max_zoom, 1 + max_zoom];
  - elimina al azar una fracción de fotogramas entre 0 y max_eliminacion (la seña
    queda más corta, como si se hubiera hecho un poco más rápido);
  - opcionalmente (--espejo) espeja las copias de índice par, lo que convierte una
    seña diestra en zurda.

Los parámetros dependen solo de --semilla, del nombre del video y del número de
copia, así que el resultado es reproducible; se registran en
<salida>/aumento_parametros.csv. La carpeta de salida nunca se borra.

--main-folder-path puede ser la carpeta de una clase (con videos) o una partición
con una subcarpeta por clase (por ejemplo rawdata_clean/train). Aumente solo el
conjunto de entrenamiento, nunca el de prueba.
"""
import argparse
import concurrent.futures
import csv
import os
import random
import shutil
import time

import cv2

from lspy_common import is_augmented, set_seed

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')


def augmentation_params(video_clip_name, i, opt):
    """Parámetros de la copia i del video (deterministas para una misma semilla)."""
    rng = random.Random('%d-%s-%d' % (opt.semilla, video_clip_name, i))
    return {
        'espejo': bool(opt.espejo and i % 2 == 0),
        'angulo': round(rng.uniform(-opt.max_rotacion, opt.max_rotacion), 2),
        'zoom': round(rng.uniform(1 - opt.max_zoom, 1 + opt.max_zoom), 3),
        'fraccion_eliminada': round(rng.uniform(0, opt.max_eliminacion), 3),
        'semilla_fotogramas': rng.randrange(2 ** 31),
    }


def augment_and_save_frames(video_path, path_of_video_to_save, params):
    """Lee cada cuadro del video, lo aumenta y lo escribe en el video de salida.
    Devuelve (fotogramas leídos, fotogramas escritos)."""
    video_reader = cv2.VideoCapture(video_path)
    fps = video_reader.get(cv2.CAP_PROP_FPS) or 30
    w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # rotación y zoom alrededor del centro, en una sola transformación afín
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), params['angulo'], params['zoom'])
    drop_rng = random.Random(params['semilla_fotogramas'])

    fourcc = 'mp4v'  # output video codec
    video_writer = cv2.VideoWriter(path_of_video_to_save, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    read = written = 0
    try:
        while video_reader.isOpened():
            ret, frame = video_reader.read()
            if not ret:
                break
            read += 1
            if drop_rng.random() < params['fraccion_eliminada']:
                continue
            if params['espejo']:
                frame = cv2.flip(frame, 1)
            frame = cv2.warpAffine(frame, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            video_writer.write(frame)
            written += 1
    finally:
        video_reader.release()
        video_writer.release()
    return read, written


def augment_video(video_path, output_folder_path, i, opt):
    video_clip_name = os.path.basename(video_path)
    # nombre de la copia: sin espacios, como en la versión anterior ('x - Trim.mp4' -> 'x-Trim_0.mp4')
    stem, ext = os.path.splitext(video_clip_name.replace(' ', ''))
    path_of_video_to_save = os.path.join(output_folder_path, '%s_%d%s' % (stem, i, ext))
    if os.path.exists(path_of_video_to_save) and not opt.sobrescribir:
        return None
    params = augmentation_params(video_clip_name, i, opt)
    read, written = augment_and_save_frames(video_path, path_of_video_to_save, params)
    return [os.path.relpath(video_path, opt.main_folder_path), os.path.basename(path_of_video_to_save),
            params['espejo'], params['angulo'], params['zoom'], params['fraccion_eliminada'], read, written]


def find_videos(folder, include_augmented):
    """Devuelve [(ruta_video, carpeta_de_salida_relativa)] de una clase o de una partición."""
    entries = sorted(os.listdir(folder))
    subdirs = [e for e in entries if os.path.isdir(os.path.join(folder, e))]
    groups = [(os.path.join(folder, d), d) for d in subdirs] if subdirs else [(folder, '')]
    videos, skipped = [], 0
    for class_dir, rel in groups:
        for name in sorted(os.listdir(class_dir)):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in VIDEO_EXTENSIONS:
                continue
            if is_augmented(stem) and not include_augmented:
                skipped += 1
                continue
            videos.append((os.path.join(class_dir, name), rel))
    return videos, skipped


if __name__ == '__main__':
    time_of_code = time.time()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--main-folder-path', type=str, required=True,
                        help='Carpeta de una clase o partición con una subcarpeta por clase')
    parser.add_argument('--output-folder-path', type=str, required=True,
                        help='Carpeta donde se escriben las copias (no se borra lo que ya contiene)')
    parser.add_argument('--max-clips', type=int, required=True, help='Copias aumentadas por video original')
    parser.add_argument('--max-rotacion', type=float, default=30, help='Ángulo máximo de rotación, en grados')
    parser.add_argument('--max-zoom', type=float, default=0.15, help='Zoom máximo (0.15 = entre 85 %% y 115 %%)')
    parser.add_argument('--max-eliminacion', type=float, default=0.2,
                        help='Fracción máxima de fotogramas eliminados al azar')
    parser.add_argument('--espejo', action='store_true', help='Espejar horizontalmente las copias de índice par')
    parser.add_argument('--copiar-originales', action='store_true',
                        help='Copiar también los originales a la salida (para usarla directamente con handtrack.py)')
    parser.add_argument('--incluir-aumentados', action='store_true',
                        help='Aumentar también los videos que ya son copias (<nombre>_<i>)')
    parser.add_argument('--sobrescribir', action='store_true', help='Reemplazar copias que ya existen')
    parser.add_argument('--semilla', type=int, default=42)
    parser.add_argument('--hilos', type=int, default=4)
    opt = parser.parse_args()
    set_seed(opt.semilla)

    if os.path.abspath(opt.main_folder_path) == os.path.abspath(opt.output_folder_path):
        parser.error('La carpeta de salida debe ser distinta de la de entrada.')

    videos, skipped = find_videos(opt.main_folder_path, opt.incluir_aumentados)
    print('Videos a aumentar: %d (se omiten %d que ya son copias aumentadas)' % (len(videos), skipped))

    jobs = []
    for video_path, rel in videos:
        out_dir = os.path.join(opt.output_folder_path, rel)
        os.makedirs(out_dir, exist_ok=True)
        if opt.copiar_originales:
            target = os.path.join(out_dir, os.path.basename(video_path))
            if not os.path.exists(target):
                shutil.copy2(video_path, target)
        jobs += [(video_path, out_dir, i) for i in range(opt.max_clips)]

    log_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=opt.hilos) as executor:
        futures = [executor.submit(augment_video, v, o, i, opt) for v, o, i in jobs]
        for n, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            row = future.result()
            if row is not None:
                log_rows.append(row)
                print('[%d/%d] %s: ángulo %.1f°, zoom %.2f, %d de %d fotogramas%s'
                      % (n, len(jobs), row[1], row[3], row[4], row[7], row[6], ', espejo' if row[2] else ''))

    log_path = os.path.join(opt.output_folder_path, 'aumento_parametros.csv')
    new_file = not os.path.exists(log_path)
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(['video_original', 'copia', 'espejo', 'angulo', 'zoom', 'fraccion_eliminada',
                             'fotogramas_original', 'fotogramas_copia'])
        writer.writerows(sorted(log_rows))

    print('Copias generadas: %d (ya existían %d). Parámetros en %s'
          % (len(log_rows), len(jobs) - len(log_rows), log_path))
    print('Tiempo total: %.1f s' % (time.time() - time_of_code))
