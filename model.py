from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import PIL.Image

train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

x_train = train_datagen.flow_from_directory("data/train",target_size = (64,64),batch_size = 32,class_mode = "categorical")
x_test = test_datagen.flow_from_directory("data/test",target_size = (64,64),batch_size = 32,class_mode = "categorical")

x_train.class_indices

#MODEL BUILDING

model = Sequential()

model.add(Convolution2D(32,(3,3),input_shape = (64,64,3),activation = "relu"))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Convolution2D(32,(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) # ANN Input...

#Adding Dense Layers

model.add(Dense(units = 128,kernel_initializer = "random_uniform",activation = "relu"))

model.add(Dense(units = 128,kernel_initializer = "random_uniform",activation = "relu"))

model.add(Dense(units = 128,kernel_initializer = "random_uniform",activation = "relu"))

model.add(Dense(units = 128,kernel_initializer = "random_uniform",activation = "relu"))

model.add(Dense(units = 128,kernel_initializer = "random_uniform",activation = "relu"))

model.add(Dense(units = 6,kernel_initializer = "random_uniform",activation = "softmax"))

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(
    x_train,
    steps_per_epoch=len(x_train),
    epochs=9,
    validation_data=x_test,
    validation_steps=len(x_test)
)

#Saving Model.
model.save('ecg_model.h5')