import csv
import numpy as np
import random
import glob
import os.path
import sys
import operator
import threading
from keras.preprocessing import image as Img
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
from tqdm import tqdm

from keras.utils import to_categorical
class DataSet():

    def __init__(self, seq_length=40, class_limit=None, image_shape=(224, 224, 3)):
        self.seq_length = seq_length
        self.class_limit = class_limit
        self.sequence_path = os.path.join('data', 'sequences')
        self.max_frames = 300  # max number of frames a video can have for us to use it
        self.data = self.get_data()
        self.classes = self.get_classes()
        self.data = self.clean_data()
        self.image_shape = image_shape

    @staticmethod
    def get_data():
        with open(os.path.join('data', 'data_file.csv'), 'r') as fin:
            reader = csv.reader(fin)
            #print(list(reader))
            data = list(reader)
            print('Retornando data para clase dataset')
            print(data)
        return data

    def clean_data(self):
        data_clean = []
        print('Data antes del clean')
        print(self.data)
        for item in self.data:
            if int(item[3]) >= self.seq_length and int(item[3]) <= self.max_frames \
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

        X, y = [], []
        for row in data:
            sequence = self.get_extracted_sequence(data_type, row)
            if sequence is None:
                print("Can't find sequence. Did you generate them?")
                raise
            X.append(sequence)
            y.append(self.get_class_one_hot(row[1]))
        return np.array(X), np.array(y)

    def get_extracted_sequence(self, data_type, sample):
        filename = sample[2]
        print('Aca hay seqlength')
        print(self.seq_length)
        path = os.path.join(self.sequence_path, filename + '-' + str(self.seq_length) + \
            '-' + data_type + '.npy')
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
        filenames."""
        path = os.path.join('data', sample[0], sample[1])
        print('Path donde busca frames')
        print(path)
        filename = sample[2]
        images = sorted(glob.glob(os.path.join(path, filename + '*jpg')))
        return images

    @staticmethod
    def rescale_list(input_list, size):
        print('Longitud de input list')
        print(len(input_list))
        assert len(input_list)>= size
        skip = len(input_list) // size
        output = [input_list[i] for i in range(0, len(input_list), skip)]
        return output[:size]

# Get the dataset.
#seq_length = 40
data = DataSet(seq_length=40, class_limit=3)
print('The data is ')
print(data.data)
base_model = InceptionV3(
    weights='imagenet',
    include_top=True
)
# We'll extract features at the final pool layer.
model = Model(
    inputs=base_model.input,
    outputs=base_model.get_layer('avg_pool').output
)

# Loop through data.
pbar = tqdm(total=len(data.data))
for video in data.data:
    print('Video es')
    print(video)
    # Get the path to the sequence for this video.
    print('En la linea 153 hay seqlength')
    print(data.seq_length)
    path = os.path.join('data', 'sequences', video[2] + '-' + str(data.seq_length) + \
        '-features')  # numpy will auto-append .npy
    print('Path es: ')
    print(path)
    # Check if we already have it.
    if os.path.isfile(path + '.npy'):
        print('Ya existe')
        pbar.update(1)
        continue

    # Get the frames for this video.
    frames = data.get_frames_for_sample(video)
    print('Frames')
    print(frames)

    # Now downsample to just the ones we need.
    frames = data.rescale_list(frames, 40)
    #print(frames)
    #extracting features and appending to build the sequence.
    sequence = []
    for image in frames:
        img = Img.load_img(image, target_size=(299, 299))
        x = Img.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        sequence.append(features[0])

    # Save the sequence.
    np.save(path, sequence)

    pbar.update(1)

pbar.close()