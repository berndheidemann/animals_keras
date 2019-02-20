import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from smallvggnet import SmallVGGNet
from keras.callbacks import ModelCheckpoint
from keras.applications.imagenet_utils import preprocess_input
import os
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import to_categorical
from PlotLoss import PlotLosses

image_size=96
batch_size = 100
num_classes = 3
epochs=50


def getMyModel():
        model = Sequential()
        model.add(Conv2D(32, (7, 7), strides=(1, 1), name='conv0', input_shape=(image_size, image_size, 3)))
        model.add(BatchNormalization(axis=3, name='bn0'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), name='max_pool'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), name="conv1"))
        model.add(Activation('relu'))
        model.add(AveragePooling2D((3, 3), name='avg_pool'))
        model.add(Flatten())
        model.add(Dense(500, activation="relu", name='rl'))
        model.add(Dropout(0.8))
        model.add(Dense(num_classes, activation='softmax', name='sm'))
        return model

def loadSet(dataset):
        classes=["cats", "dogs", "panda"]

        nb_samples=0
        files=[]
        for i,cls in enumerate(classes):
                files.append(os.listdir("animals/"+dataset+"/"+cls))
                nb_samples+=len(files[i])
        x_train=np.zeros((nb_samples, image_size, image_size, 3))
        y_train=np.zeros((nb_samples))

        c=0
        for j,cls in enumerate(classes):
                for file in files[j]:
                        img = image.load_img("./animals/" + dataset + "/" + cls + "/" + file, target_size=(image_size, image_size, 3))
                        x = image.img_to_array(img)
                        x = preprocess_input(x)
                        x_train[c]=x
                        y_train[c]=j
                        c+=1

        return (x_train, y_train)

(x_train, y_train)=loadSet("train")
(x_test, y_test)=loadSet("test")

y_train=to_categorical(y_train, num_classes=3)
y_test=to_categorical(y_test, num_classes=3)


train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preprocess_input)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

model=getMyModel()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint('./saves/model-{epoch:02d}-{acc:.4f}.h5', verbose=1, monitor='val_acc',save_best_only=True, mode='auto')

model.fit_generator(
        train_datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=20000 // batch_size,
        epochs=epochs,
        validation_data=test_datagen.flow(x_test, y_test, batch_size=50),
        callbacks=[checkpoint, PlotLosses(slowlyCutBeginning=False)],
        validation_steps=800 // batch_size)
model.save('myNet.h5')  # always save your weights after training or during training

score=model.evaluate_generator(test_datagen(x_test, y_test), steps=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])