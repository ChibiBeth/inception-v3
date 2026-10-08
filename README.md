# MEDIAPIPE, INCEPTION V3, RNN FOR SIGN RECOGNITION


## Installation

Make sure you have Python (https://www.python.org/) installed, then install Tensorflow (https://www.tensorflow.org/install/) on your system, and clone this repo. <br/>
Then install the requirements.


```commandline
pip install -r requirements.txt
```

## Usage
### Data Structure
```
raw_data
└───train
│   └───miercoles
│       │   file111.mp4
│       │   file112.mp4
│       │   ...
│   
└───test
│       └───miercoles
│        │   file021.mp4
│        │   file022.mp4
```
### Data Augmentation

Generates copies of each original training video with random rotation, zoom and frame dropping (add `--espejo` to also mirror even copies). It is reproducible (`--semilla`), logs every copy's parameters and never deletes the output folder:
```commandline
python data_augmentation.py --main-folder-path rawdata/train --output-folder-path rawdata_aug/train --max-clips 4 --espejo --copiar-originales
```

### PreProcessing

Detects the hands with MediaPipe on the full (undistorted) frames and resamples each video to a fixed number of frames. It saves the hand drawings and the landmark coordinates:
```commandline
python handtrack.py -i rawdata_aug -o data
```
### Generating INCEPTION V3 feature sequences
Extracts 2048 ImageNet features per frame (global average pooling):
```commandline
python extract_features_harp.py
```
### Retraining the last layer of INCEPTION V3 (optional)
Trains a new softmax layer on top of the frozen Inception V3 features, giving a per-frame probability for each sign:
```commandline
python retrain_inception.py
```
### Training the RNN Model
```commandline
python train_lstm_harp.py                                       # --data-type features2048 | probs | landmarks, --arch ligera | original
```
### Test the predict process
```commandline
python predict_harp.py ROJO.mp4
```
### Metrics
```commandline
python evaluate_metrics.py --model lstm_senha_model --raw-dir rawdata_aug --signers-csv personas_video.csv
```

Full documentation (in Spanish) is in [docs/](docs/README.md).

## Credits and origin of the code

This project was not written from scratch. It started from two MIT-licensed repositories, whose copyright notices are kept in [LICENSE](LICENSE):

- **[hthuwal/sign-language-gesture-recognition](https://github.com/hthuwal/sign-language-gesture-recognition)** (Harish Chandra Thuwal), the code of Masood, Srivastava, Thuwal and Ahmad (2018), *Real-Time Sign Language Gesture (Word) Recognition from Video Sequences Using CNN and RNN*, doi:[10.1007/978-981-10-7566-7_63](https://doi.org/10.1007/978-981-10-7566-7_63). The two per-frame representations compared in the thesis (softmax probabilities of a retrained Inception V3 vs. the 2048 values of its last pooling layer) follow its two approaches. `loadpicklefileanddisplay.py` comes from this repository. Its authors ask to cite the paper if the project is useful.
- **[harvitronix/five-video-classification-methods](https://github.com/harvitronix/five-video-classification-methods)** (Matt Harvey). The `DataSet` class in `extract_features_harp.py`, the organisation of videos as frame sequences and the `--arch original` LSTM derive from it. The `_harp` suffix comes from *human activity recognition project*, the name of the authors' first adaptation of this code.

Everything else (MediaPipe preprocessing, data augmentation, retrained Inception head, light LSTM, grouped validation, evaluation, multi-seed experiments and the similar-sign analysis) was written for the thesis.

## Data

The original videos are **not** included and are not shared: participants authorised their use for the thesis and derived academic publications only. If a dataset is released, it will contain only preprocessed data (hand-skeleton drawings, MediaPipe coordinates and Inception V3 features) with pseudonymous signer codes (P1–P10).

## Thesis version

The results reported in the thesis correspond to tag `tesis-v1.0` (commit `6f1e1401debd2485420a6d900e170b896473e96a`).
