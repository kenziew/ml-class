from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.utils import np_utils
from wandb.keras import WandbCallback
import wandb
import os

run = wandb.init()
config = run.config
config.first_layer_convs = 32 # number of convolution in first layer
config.first_layer_conv_width = 3 #specifying size of kernel so 3x3
config.first_layer_conv_height = 3
config.dropout = 0.2
config.dense_layer_size = 128
config.img_width = 28
config.img_height = 28
config.epochs = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# normalize data
X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

# X_train shape 60000, 28 by 28 
# reshape input data
X_train = X_train.reshape(
    X_train.shape[0], config.img_width, config.img_height, 1)
X_test = X_test.reshape(
    X_test.shape[0], config.img_width, config.img_height, 1)
# dimension of X_train after reshape is now 60000, 1

# one hot encode outputs
# taking list of 0 to 9 and encoding it
y_train = np_utils.to_categorical(y_train) #max number in any cell of y_train matrix= 1
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
labels = range(10)

# build model
model = Sequential()
model.add(Conv2D(32,
                 (config.first_layer_conv_width, config.first_layer_conv_height), #kernel size 3 by 3 so it is a 3 by 3 convolution
                 input_shape=(28, 28, 1), #the new channel dimension
                 activation='relu')) 
# shape of data coming out of the convolution layer 26 by 26 by 32. 32 convolutions so output 32 feature maps. 
# when you do a convolution, you lose dimensions. with 3 by 3 you lose 2 pixels. can fix this with zero padding within same function
model.add(MaxPooling2D(pool_size=(2, 2))) # shape of data comingout 13 by 13 by 32 cause resize in half
model.add(Dropout(0.4))
model.add(Conv2D(64, (3,3), activation="relu")) #added. twice the number of feature maps but convolving
model.add(MaxPooling2D(pool_size=(2, 2))) #added
#added below dropout line after running it and seeing val acc < acc so overfitting occurred
model.add(Dropout(0.4)) #drop 40% of values in a feature map so the network doesnt use it
model.add(Flatten()) # shape after flattening is 13*13*32=5408 numbers
model.add(Dense(config.dense_layer_size, activation='relu')) # dense layer has 5408*128+bias terms so 692352 learned parameters
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])


model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=config.epochs,
          callbacks=[WandbCallback(data_type="image", save_model=False)])



#if  validation acc is less than than acc so need to add drop out. when this occurs it means there is over fitting. add dropout or reduce the number of parameters in our network