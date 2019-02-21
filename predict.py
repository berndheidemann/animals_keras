
from keras.models import load_model
import argparse
import pickle
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
from dataLoader import loadSet
from dataLoader import classes
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input

image_size=96
batch_size = 50


(x_test, y_test)=loadSet("test")

model = load_model("./saves/model-10-0.8239.h5")
preds = model.predict(x_test)

ax = []
columns = 5
rows = 5
fig = plt.figure(figsize=(9, 13))

for j in range( columns*rows ):
    i=np.random.randint(0, x_test.shape[0])
    image = x_test[i]+127
    title= classes[np.argmax(preds[i],axis=0)]
    ax.append( fig.add_subplot(rows, columns, j+1) )
    ax[-1].set_title(title)
    plt.imshow(image.astype('uint8'))

plt.show()