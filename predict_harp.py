import numpy as np
import os.path
from keras.preprocessing import image as Img
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import glob
import cv2


def rescale_list(input_list, size):
    assert len(input_list) >= size
    skip = len(input_list) // size
    output = [input_list[i] for i in range(0, len(input_list), skip)]
    return output[:size]

model = load_model("lstm_senha_model")
classes =  glob.glob(os.path.join('data','train', '*'))
classes = [classes[i].split('\\')[2] for i in range(len(classes))]
classes = sorted(classes)
print(classes)

image_name = 'soltero_hugo.avi'
cam = cv2.VideoCapture(image_name) 
currentframe = 0
  
frames=[]
while(True): 
    ret,frame = cam.read() 
    if ret: 
        # if video is still left continue creating images 
        name = 'testFinal/frame'+'9' +"frame_no"+ str(currentframe) + '.jpg'
        cv2.imwrite(name, frame) 
        print(frame)
        frames.append(name)  
        currentframe += 1
    else: 
        break
cam.release() 
cv2.destroyAllWindows()
print(frames)
rescaled_list = rescale_list(frames,40)

base_model = InceptionV3(
    weights='imagenet',
    include_top=True
)
# We'll extract features at the final pool layer.
inception_model = Model(
    inputs=base_model.input,
    outputs=base_model.get_layer('avg_pool').output
)
sequence = []
for image in rescaled_list:
        img = Img.load_img(image, target_size=(299, 299))
        x = Img.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = inception_model.predict(x)
        sequence.append(features[0])

sequence = np.array([sequence])
prediction = model.predict(sequence)
maxm = prediction[0][0]
maxid = 0
for i in range(len(prediction[0])):
      if(maxm<prediction[0][i]):
            maxm = prediction[0][i]
            maxid = i
#print(frames)
print(image_name,' ------- ',classes[maxid])