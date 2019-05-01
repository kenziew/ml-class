from keras.datasets import mnist
from keras.models import Sequential # meaning each layer feeds into next layer
from keras.layers import Dense, Flatten # 2 types of layers

import wandb #allows us to see results as we train
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

is_five_train = y_train == 5 # 0 when not 5, and 1 when it is a 5
is_five_test = y_test == 5
labels = ["Not Five", "Is Five"]

img_width = X_train.shape[1]
img_height = X_train.shape[2]

# create model # first shallow learning model
model = Sequential()
model.add(Flatten(input_shape=(img_width, img_height))) # add intitial flattened layer. takes the 2D 28 by 28 pixels and turns it into 784 lenght array
model.add(Dense(1,activation="sigmoid")) # dense bc densely connected to previous layer. every input has learned weight from inputs of previously layer. 1 is because we want 1 output. if 2 it would output 2 numbers. so this is a single perceptron. so 1 means you are ouputting the 'schematic of rosenblatts perceptron
# shape of data after dense 1 is just 1 
model.compile(loss='mse', optimizer='adam',
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, is_five_train, epochs=6, validation_data=(X_test, is_five_test),
          callbacks=[WandbCallback(data_type="image", labels=labels, save_model=False)])
model.save('perceptron.h5') # saves weights so it can then be used for inference 
