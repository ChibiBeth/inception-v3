"""Utilidades compartidas por los scripts del pipeline (sin dependencias pesadas)."""
import glob
import json
import os
import random
import re
from collections import defaultdict

import numpy as np

# Sufijo que data_augmentation.py agrega a cada copia: '<original>_<i>'
AUG_SUFFIX = re.compile(r"_\d$")
TRIM = re.compile(r"\s*-\s*Trim", re.IGNORECASE)
ORIGINAL_TRIM = re.compile(r"\s-\sTrim", re.IGNORECASE)

# Representaciones que puede consumir la LSTM (data_type de los .npy en data/sequences)
DATA_TYPES = {
    'features2048': 'Inception V3 (ImageNet), promedio global: 2048 valores por fotograma',
    'probs': 'Inception V3 con la última capa reentrenada: probabilidad de cada seña por fotograma',
    'landmarks': 'Coordenadas de MediaPipe: 2 manos x 21 puntos x (x, y, z) = 126 valores por fotograma',
    'features': '(versión anterior) proyección aleatoria de 10 valores; solo para comparar',
}


def set_seed(seed=42):
    """Fija las semillas de Python, NumPy y TensorFlow (si está importado)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def original_stem(stem):
    """'agosto_cr1_0' -> 'agosto_cr1';  'marzo_cr4-Trim_2' -> 'marzo_cr4'.

    data_augmentation.py elimina los espacios del nombre, por lo que un nombre con
    ' - Trim' (con espacios) siempre es un video original: 'marzo_cr1_2 - Trim'
    -> 'marzo_cr1_2' (no se interpreta el '_2' como sufijo de aumento).
    """
    if ORIGINAL_TRIM.search(stem):
        return TRIM.sub("", stem).strip()
    s = TRIM.sub("", stem).strip()
    while AUG_SUFFIX.search(s):
        s = AUG_SUFFIX.sub("", s)
    return s


def is_augmented(stem):
    return TRIM.sub("", stem).strip() != original_stem(stem)


def video_original(split, stem):
    """Video original de una fila de data_file.csv. Solo train tiene copias aumentadas:
    en test, un nombre como 'marzo_cr1_2' es un original y no se recorta."""
    return original_stem(stem) if split == 'train' else TRIM.sub("", stem).strip()


def is_augmented_row(split, stem):
    return split == 'train' and is_augmented(stem)


def sequence_file(data_dir, split, stem, seq_length, data_type):
    """data/sequences/<partición>/<video>-<T>-<tipo>.npy. La subcarpeta por partición
    evita que una copia de train ('marzo_cr1' -> 'marzo_cr1_2') pise a un original de
    test con el mismo nombre."""
    return os.path.join(data_dir, 'sequences', split, '%s-%d-%s.npy' % (stem, seq_length, data_type))


def frame_pattern(stem):
    """Patrón glob de los fotogramas de UN video: '<stem>-NNNN.jpg' (y el relleno
    '<stem>_NNNN.jpg' de la versión anterior de handtrack.py). Evita tomar los
    fotogramas de las copias '<stem>_0', '<stem>_1'... (H2)."""
    return glob.escape(stem) + '[-_][0-9][0-9][0-9][0-9].jpg'


def grouped_train_val_split(stems, labels, val_frac=0.2, seed=42):
    """Separa una validación desde train agrupando por video original (H3).

    Todas las copias aumentadas de un original quedan del mismo lado, y se
    reserva al menos un original por clase para validación (estratificado).
    Devuelve (indices_train, indices_val).
    """
    by_class = defaultdict(set)
    for stem, label in zip(stems, labels):
        by_class[label].add(original_stem(stem))
    rng = random.Random(seed)
    val_groups = set()
    for label in sorted(by_class):
        groups = sorted(by_class[label])
        rng.shuffle(groups)
        n_val = max(1, int(round(val_frac * len(groups)))) if len(groups) > 1 else 0
        val_groups.update(groups[:n_val])
    tr_idx = [i for i, s in enumerate(stems) if original_stem(s) not in val_groups]
    va_idx = [i for i, s in enumerate(stems) if original_stem(s) in val_groups]
    return tr_idx, va_idx


def meta_path(model_path):
    return model_path.rstrip('/\\') + '.json'


def save_meta(model_path, meta):
    """Guarda junto al modelo la lista de clases y cómo se construyó la entrada (H10)."""
    with open(meta_path(model_path), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_meta(model_path):
    p = meta_path(model_path)
    if not os.path.isfile(p):
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)
