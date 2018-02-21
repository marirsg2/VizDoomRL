
import pickle

from keras.layers import Input, Dense, Conv2D, MaxPooling2D,Flatten,Dropout
from keras.models import Model,Sequential
from keras.losses import mean_squared_logarithmic_error, mean_squared_error
from keras import backend as K
from keras.callbacks import TensorBoard
import numpy as np

#We really want mean squared error for score prediction, with
# Q(s,a). need conv+dense -> score. The action is a PARALLEL input, one hot
# encoded. This after the dense network and before the second dense network
# that calculates the score. The first dense network gives values from visual features
# to calculate the reward. the second dense network uses the visual feature values
# and the action one hot encoding to predict score.

#training params
img_rows,img_cols = 125,200
batch_size = 128
epochs = 60

viz_doom_images = []
viz_doom_action = []
viz_doom_reward = []
with open("vizdoom_data.p","rb") as vizdoom_data:
    viz_doom_images = pickle.load(vizdoom_data)
    viz_doom_action = pickle.load(vizdoom_data)
    viz_doom_reward = pickle.load(vizdoom_data)

#clean up the action format
action_data = [x.index(True) for x in viz_doom_action]
viz_doom_action = np.array(action_data)
viz_doom_reward = np.array(viz_doom_reward)
viz_doom_reward = np.abs(viz_doom_reward/100.0)

print np.max(viz_doom_reward)


train_data_end_index = int(0.8*len(viz_doom_action))
#train_data_end_index = int(1.0*len(viz_doom_action))
#now split
train_image_data = np.array(viz_doom_images[:train_data_end_index])
train_reward_data = np.array(viz_doom_reward[:train_data_end_index])

# test_image_data = train_image_data
# test_reward_data = train_reward_data
test_image_data = np.array(viz_doom_images[train_data_end_index:])
test_reward_data = np.array(viz_doom_reward[train_data_end_index:])

#shape the data as needed
train_image_data = train_image_data.reshape(train_image_data.shape[0],1,img_rows,img_cols)
test_image_data = test_image_data.reshape(test_image_data.shape[0],1,img_rows,img_cols)
input_shape = (1,125, 200)


#todo first do the DQN approach. albeit with a single frame. for 3 actions
#just dump all data, the reward will be higher when the monster is in front
#of gun and zero otherwise.

#-----------------------------------------------------



model = Sequential()
model.add(Conv2D(32, kernel_size=(6, 6), strides=(3,3),
                 padding="same",
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3),strides=(2,2), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),padding="same"))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(1, activation=None))

#mean squared error works better than the absolute error
model.compile(loss= mean_squared_error,
              optimizer="adagrad") #??adagrad is supposed to be better for infreq data

model.fit(train_image_data, train_reward_data,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_image_data, test_reward_data))

score = model.evaluate(test_image_data, test_reward_data, verbose=0)

print(score)

print model.predict(test_image_data[160:180]) , test_reward_data[160:180]
print model.predict(train_image_data[770:780]) , train_reward_data[770:780]

#todo, test for the image when the reward is 100 (high), also test -1 and -6 case