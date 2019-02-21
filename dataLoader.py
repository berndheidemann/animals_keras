import os
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input

classes = ["cats", "dogs", "panda"]


def loadSet(dataset, image_size=96, preprocess=True):

    nb_samples = 0
    files = []
    for i, cls in enumerate(classes):
        files.append(os.listdir("animals/" + dataset + "/" + cls))
        nb_samples += len(files[i])
        print(cls + " nb:" +str(len(files[i])))
    x_train = np.zeros((nb_samples, image_size, image_size, 3))
    y_train = np.zeros((nb_samples))

    c = 0
    for j, cls in enumerate(classes):
        for file in files[j]:
            img = image.load_img("./animals/" + dataset + "/" + cls + "/" + file,
                                 target_size=(image_size, image_size, 3))
            x = image.img_to_array(img)
            if preprocess:
                x = preprocess_input(x)
            x_train[c] = x
            y_train[c] = j
            c += 1

    return (x_train, y_train)