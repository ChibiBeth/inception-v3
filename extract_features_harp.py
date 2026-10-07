import argparse
import csv
import glob
import os.path

import cv2
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm

from lspy_common import frame_pattern, sequence_file, set_seed


class DataSet():

    def __init__(self, seq_length=150, class_limit=None, image_shape=(299, 299, 3), max_frames=None):
        self.seq_length = seq_length
        self.class_limit = class_limit
        self.sequence_path = os.path.join('data', 'sequences')
        self.max_frames = max_frames  # máximo de fotogramas por video (None = sin límite)
        self.data = self.get_data()
        self.classes = self.get_classes()
        self.data = self.clean_data()
        self.image_shape = image_shape

    @staticmethod
    def get_data():
        with open(os.path.join('data', 'data_file.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data = [row for row in reader if len(row) >= 4]
        return data

    def clean_data(self):
        data_clean = []
        for item in self.data:
            if int(item[3]) >= self.seq_length \
                    and (self.max_frames is None or int(item[3]) <= self.max_frames) \
                    and item[1] in self.classes:
                data_clean.append(item)

        return data_clean

    def get_classes(self):
        classes = []
        for item in self.data:
            if item[1] not in classes:
                classes.append(item[1])
        classes = sorted(classes)
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes

    def get_class_one_hot(self, class_str):
        # Encode it first.
        label_encoded = self.classes.index(class_str)
        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.classes))
        assert len(label_hot) == len(self.classes)
        return label_hot

    def split_train_test(self):
        train = []
        test = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test

    def get_all_sequences_in_memory(self, train_test, data_type):
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test
        print("Loading %d samples into memory for %sing." % (len(data), train_test))
        return self.load_rows(data, data_type)

    def load_rows(self, rows, data_type):
        X, y = [], []
        for row in rows:
            sequence = self.get_extracted_sequence(data_type, row)
            if sequence is None:
                raise FileNotFoundError("Can't find sequence %s. Did you generate them?"
                                        % self.sequence_file(row, data_type))
            X.append(sequence)
            y.append(self.get_class_one_hot(row[1]))
        return np.array(X), np.array(y)

    def sequence_file(self, sample, data_type):
        return sequence_file('data', sample[0], sample[2], self.seq_length, data_type)

    def get_extracted_sequence(self, data_type, sample):
        path = self.sequence_file(sample, data_type)
        if os.path.isfile(path):
            return np.load(path)
        else:
            return None

    def get_frames_by_filename(self, filename, data_type):
        sample = None
        for row in self.data:
            if row[2] == filename:
                sample = row
                break
        if sample is None:
            raise ValueError("Couldn't find sample: %s" % filename)
        sequence = self.get_extracted_sequence(data_type, sample)
        if sequence is None:
            raise ValueError("Can't find sequence. Did you generate them?")
        return sequence

    @staticmethod
    def get_frames_for_sample(sample):
        """Given a sample row from the data file, get all the corresponding frame
        filenames (only this video's frames, not those of its augmented copies)."""
        path = os.path.join('data', sample[0], sample[1])
        # '-' (fotogramas) se ordena antes que '_' (relleno de la versión anterior)
        images = sorted(glob.glob(os.path.join(path, frame_pattern(sample[2]))))
        return images

    @staticmethod
    def rescale_list(input_list, size):
        """Elige `size` elementos equiespaciados de toda la lista."""
        assert len(input_list) >= size
        idx = np.linspace(0, len(input_list) - 1, num=size).round().astype(int)
        return [input_list[i] for i in idx]


def build_extractor():
    """Inception V3 preentrenada en ImageNet, sin la capa de clasificación y con
    promedio global: 2048 características por imagen. No tiene pesos aleatorios,
    por lo que entrenamiento y predicción producen exactamente los mismos
    vectores (H1)."""
    from keras.applications.inception_v3 import InceptionV3
    return InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_shape=(299, 299, 3))


def load_frames(paths):
    """Lee los .jpg de un video como arreglo RGB (n, 299, 299, 3)."""
    return np.stack([cv2.cvtColor(cv2.resize(cv2.imread(p), (299, 299)), cv2.COLOR_BGR2RGB) for p in paths])


def canvases_to_rgb(canvases):
    """Mismo preprocesamiento que en el entrenamiento para dibujos en memoria:
    se pasan por JPEG (como al guardarlos en handtrack.py) y se convierten a RGB (H10)."""
    out = []
    for img in canvases:
        ok, buf = cv2.imencode('.jpg', img)
        out.append(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    return np.stack(out)


def extract_features(model, rgb_frames, batch_size=32):
    """(n, 299, 299, 3) RGB -> (n, 2048), procesando los fotogramas por lotes (H14)."""
    from keras.applications.inception_v3 import preprocess_input
    x = preprocess_input(rgb_frames.astype(np.float32))
    # model(...) en lugar de model.predict(...): predict dentro de un bucle acumula memoria en TF 2.15
    return np.concatenate([model(x[i:i + batch_size], training=False).numpy()
                           for i in range(0, len(x), batch_size)])


def main():
    parser = argparse.ArgumentParser(description='Extrae 2048 características por fotograma con Inception V3.')
    parser.add_argument('--seq-length', type=int, default=150)
    parser.add_argument('--class-limit', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    set_seed(42)
    data = DataSet(seq_length=args.seq_length, class_limit=args.class_limit)
    model = build_extractor()

    # Loop through data.
    for video in tqdm(data.data, desc='Extrayendo características'):
        # Get the path to the sequence for this video.
        path = data.sequence_file(video, 'features2048')
        # Check if we already have it.
        if os.path.isfile(path):
            continue

        # Get the frames for this video.
        frames = data.get_frames_for_sample(video)
        if len(frames) < data.seq_length:
            tqdm.write('[aviso] %s tiene %d fotogramas (< %d); se omite.' % (video[2], len(frames), data.seq_length))
            continue

        # Now downsample to just the ones we need.
        frames = data.rescale_list(frames, data.seq_length)
        sequence = extract_features(model, load_frames(frames), args.batch_size)

        # Save the sequence.
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, sequence.astype(np.float32))


if __name__ == '__main__':
    main()
