import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from smallvggnet import SmallVGGNet
from keras.callbacks import ModelCheckpoint
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt


def getMyModel():
        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=(image_size, image_size, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        return model


image_size=96
batch_size = 100
num_classes = 3
epochs=500

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.2,        #maximal 20% Winkelveränderungen
        zoom_range=0.2,         # maximal 20% Zoom
        horizontal_flip=True,
        preprocessing_function=preprocess_input)


test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input) # only rescale for testing

train_generator = train_datagen.flow_from_directory(
        'animals/train',  # this is the target directory
        target_size=(image_size, image_size),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'animals/test',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical')



model=getMyModel()
#model=SmallVGGNet.build(image_size, image_size, 3, train_generator.num_classes)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
checkpoint = [ModelCheckpoint(filepath='models.h5', save_best_only=True)]
history=model.fit_generator(
        train_generator,
        steps_per_epoch=20000 // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=checkpoint,
        validation_steps=800 // batch_size)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.show()

model.save('myNet.h5')  # always save your weights after training or during training

score=model.evaluate_generator(validation_generator, steps=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])