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
