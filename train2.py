import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.python.keras.callbacks import EarlyStopping
from tqdm import tqdm
from tensorflow.keras.models import load_model

NUM_FRAMES = 10
IMG_SIZE = 100
LR = 0.0001
EPOCHS = 50
BATCH_SIZE = 5

cnn = load_model('./Violence_detection-CNN-BiLSTM.h5')


print('\n Loading in the data')
data = np.load('data5.npy', allow_pickle=True)
print(len(data))

X = np.array([i[0] for i in data]).reshape(-1 , 10, IMG_SIZE, IMG_SIZE, 3)
Y = np.array([i[1] for i in data])
X = X.astype('float32')/255

print(X.shape)
print(Y.shape)



print('\n Compiling Model and loading Optimizers...')
opt = tf.keras.optimizers.SGD(learning_rate=LR)
cnn.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

NAME = 'Violence_detection-CNN-BiLSTM'
earlystopping = EarlyStopping(monitor='val_accuracy', patience= 5)


print('\n Starting Training')
cnn.fit(X, Y, epochs= EPOCHS, validation_split=0.1, batch_size=BATCH_SIZE, verbose=1, callbacks= [earlystopping])
print('\n Done Training')

print("\n Saving the model.... ")
cnn.save(NAME + '.h5', overwrite=True, include_optimizer=True)
print("\n Model saved as " + NAME + '.h5')
