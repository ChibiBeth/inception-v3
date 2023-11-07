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

Run the script data_augmentation.py specifying the input and output directory, and max clips per video as below:
```commandline
python data_augmentation.py --main-folder rawdata/train/broma  --output-folder rawdata/trainaug/broma --max-clips 5
```

### PreProcessing

Run the script handtrack.py specifying the input and output directory as below:
```commandline
python handtrack.py -i raw_data -o data
```
### Retreaining and generating INCEPTION V3 model sequences
Run the script extract_features_harp.py
```commandline
python extract_features_harp.py
```
### Training the RNN Model
Run the script train_lstm_harp.py
```commandline
python train_lstm_harp.py
```
### Test the predict process
You can run a test of prediction using the script predict_harp.py
```commandline
python predict_harp.py
```


