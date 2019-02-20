
from keras.models import load_model
import argparse
import pickle
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

image_size=96
batch_size = 50

test_datagen = ImageDataGenerator(rescale=1./255) # only rescale for testing

validation_generator = test_datagen.flow_from_directory(
        'animals/test',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical')

x,y = validation_generator.next()
labels = list(validation_generator.class_indices)

model = load_model("models.h5")
preds = model.predict(x)

ax = []
columns = 5
rows = 5

fig = plt.figure(figsize=(9, 13))

for i in range( columns*rows ):
    image = x[i] * 255
    title= labels[np.argmax(preds[i],axis=0)]
    ax.append( fig.add_subplot(rows, columns, i+1) )
    ax[-1].set_title(title)
    plt.imshow(image.astype('uint8'))

plt.show()