from vizdoom import *
import os
import itertools as it
import keras
from random import sample, randint, random
import random
from time import time, sleep
from keras.models import Sequential, Model, load_model as lm
from keras.layers import Dense, Activation, Conv2D, Input, Flatten, LSTM, Reshape, RepeatVector
from keras.optimizers import Adam

import numpy as np
import skimage.color, skimage.transform
from tqdm import trange
import copy

# Q-learning hyperparams
learning_rate = 0.001
discount_factor = 0.95
epochs = 10
learning_steps_per_epoch = 2000
replay_memory_size = 10000

# NN learning hyperparams
batch_size = 1

# Training regime
test_episodes_per_epoch = 100

# Image params
resolution = (30, 45)

# Other parameters
frame_repeat = 12
resolution = [30, 45]
kframes = 1
resolution[1] = resolution[1] * kframes
episodes_to_watch = 10

model_savefile = "models/model-doom_disc50.pth"
tmp_model_savefile = "models/tmp_model.pth"

if not os.path.exists('models'):
    os.makedirs('models')

save_model = True
load_model = False
skip_learning = False

config_file_path = "../ViZDoom/scenarios/basic.cfg"


import warnings
warnings.filterwarnings("ignore")


import keras.backend as K

def get_states(model):
    return [K.get_value(s) for s,_ in model.state_updates]

def set_states(model, states):
    for (d,_), s in zip(model.state_updates, states):
        K.set_value(d, s)


def preprocess(img):
    """"""
    img = skimage.transform.resize(img, [resolution[0], resolution[1] // kframes])
    img = img.astype(np.float32)
    return img


class ReplayMemory:
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, channels, resolution[0], resolution[1] // kframes)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, 0, :, :] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, 0, :, :] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        augmented_i = [list(range(j - kframes + 1, j + 1)) for j in i]
        s1 = np.array([self.s1.take(i, mode='wrap', axis=0) for i in augmented_i])
        s1 = np.moveaxis(s1, [0, 1, 2, 3, 4], [0, 3, 1, 2, 4])
        reshape = s1.shape[0:3] + tuple([-1])
        s1 = np.reshape(s1, reshape)

        s2 = np.array([self.s2.take(i, mode='wrap', axis=0) for i in augmented_i])
        s2 = np.moveaxis(s2, [0, 1, 2, 3, 4], [0, 3, 1, 2, 4])
        reshape = s2.shape[0:3] + tuple([-1])
        s2 = np.reshape(s2, reshape)
        return s1, self.a[i], s2, self.isterminal[i], self.r[i]

    def get_last_entry(self):
        target_idx = self.capacity-1 #assume it is the last one (for looparound case)
        if (self.pos != 0):
            target_idx = self.pos-1 #previous case
        #--end if
        i = [target_idx]
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]


def create_model(available_actions_count):


    state_input = Input( batch_shape=(batch_size,1, resolution[0], resolution[1]))
    conv1 = Conv2D(8, 6, strides=3, activation='relu', data_format="channels_first")(
        state_input)  # filters, kernal_size, stride
    conv2 = Conv2D(8, 3, strides=2, activation='relu', data_format="channels_first")(
        conv1)  # filters, kernal_size, stride
    flatten = Flatten()(conv2)

    reshape_layer = RepeatVector(1)(flatten)
    lstm_layer = LSTM(32,stateful=True)(reshape_layer)
    fc1 = Dense(128,activation='relu')(lstm_layer)
    fc2 = Dense(available_actions_count)(fc1)

    model = keras.models.Model(input=state_input, output=fc2)
    adam = Adam(lr= learning_rate)
    model.compile(loss="mse", optimizer=adam)
    print(model.summary())


    #this is a copy of the model to keep weights when predicting during training
    conv1 = Conv2D(8, 6, strides=3, activation='relu', data_format="channels_first")(
        state_input)  # filters, kernal_size, stride
    conv2 = Conv2D(8, 3, strides=2, activation='relu', data_format="channels_first")(
        conv1)  # filters, kernal_size, stride
    flatten = Flatten()(conv2)
    reshape_layer = RepeatVector(1)(flatten)
    lstm_layer = LSTM(32, stateful=True)(reshape_layer)
    fc1 = Dense(128, activation='relu')(lstm_layer)
    fc2 = Dense(available_actions_count)(fc1)
    copy_model = keras.models.Model(input=state_input, output=fc2)
    adam = Adam(lr=0.001)
    copy_model.compile(loss="mse", optimizer=adam)



    return state_input, model, copy_model



def learn_from_memory(model,is_lstm = False, clone_model = None):
    """ Use replay memory to learn. Ignore s2 if s1 is terminal """


    s1 = None
    if is_lstm == True:
        s1, a, s2, isterminal, r = memory.get_last_entry()
    else:
        if memory.size > batch_size:
            s1, a, s2, isterminal, r = memory.get_sample(batch_size)


    # lstm predict updates the state of the lstm modules
    #get the current state action values. LET the lstm state get updated,
    # BUT save the weights for fitting
    model.save(tmp_model_savefile)
    target_q = model.predict(s1, batch_size=batch_size)#lstm predict updates the state of the lstm modules

    #lstm predict updates the state of the lstm modules
    #get next state lstm values, but recover the lstm state using the clone model to hold weights

    q_next = model.predict(s2, batch_size=batch_size)#lstm predict updates the state of the lstm modules
    max_q_next = np.max(q_next, axis=1)
    target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * max_q_next
    #now recover the weights (including the memory) for fitting
    model = lm(tmp_model_savefile)
    model.fit(s1, target_q,batch_size=batch_size, verbose=0)

    #todo test if fit changed the prediction

    #AND again step the lstm forward for the next state.
    _ = model.predict(s1, batch_size=batch_size)


class StateBuilder:
    def __init__(self, frame_resolution, frames_per_state=1, axis=3):
        self.pos = 0
        self.size = frames_per_state
        self.frames = np.zeros(
            (frame_resolution[0], frame_resolution[1], frame_resolution[2], self.size, frame_resolution[3]))

    def get_state(self, frame):
        self.frames[:, :, :, self.pos, :] = frame
        self.state = self.frames.reshape(*self.frames.shape[:-2],
                                         -1)  # TODO: make sure order is the same as in training!
        self.state = self.state.take(range(45 * (self.pos - self.size + 1), 45 * (self.pos + 1)), axis=3)
        self.pos = (self.pos + 1) % self.size
        return self.state


def get_best_action(state, preserve_state = False):
    if preserve_state:
        clone_model.set_weights(model.get_weights())
    q = model.predict(state, batch_size=1)
    if preserve_state:
        model.set_weights(clone_model.get_weights())
    m = np.argmax(q, axis=1)[0]
    action = m  # wrong
    return action


def perform_learning_step(epoch):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                   (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    s1 = preprocess(game.get_state().screen_buffer)

    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    if random.random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        s1 = s1.reshape([1, 1, resolution[0], resolution[1] // kframes])
        a = get_best_action(s1,preserve_state=True)
    reward = game.make_action(actions[a], frame_repeat)

    isterminal = game.is_episode_finished()
    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, isterminal, reward)

    #todo for lstms train with the stream of data. No need to randomize the input
    learn_from_memory(model, is_lstm = True,clone_model=clone_model)


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--kframes', type=int)
    parser.add_argument('-t', '--test')
    args, extras = parser.parse_known_args()
    if args.kframes:
        kframes = args.kframes
    if args.test:
        load_model = True
        skip_learning = True

    # Create Doom instance
    game = initialize_vizdoom(config_file_path)

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=replay_memory_size)

    if load_model:
        print("Loading model from: ", model_savefile)
        model = lm(model_savefile)
        pass
    else:
        my_input, model, clone_model = create_model(len(actions))

    print("Starting the training!")
    time_start = time()
    if not skip_learning:
        for epoch in range(epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []
            #todo reset lstm states
            model.reset_states()
            print("Training...")
            game.new_episode()
            # with the lstm, what you have to do is RESET the model when the single episode is over
            for learning_step in trange(learning_steps_per_epoch, leave=True):
                perform_learning_step(epoch)
                if game.is_episode_finished():
                    score = game.get_total_reward()
                    train_scores.append(score)
                    game.new_episode()
                    train_episodes_finished += 1
                    #todo we reset states here.
                    model.reset_states()

            print("%d training episodes played." % train_episodes_finished)

            train_scores = np.array(train_scores)

            print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

            print("\nTesting...")
            test_episode = []
            test_scores = []
            for test_episode in trange(test_episodes_per_epoch, leave=False):
                game.new_episode()
                # todo reset lstm states
                model.reset_states()
                sb = StateBuilder((1, 1, resolution[0], resolution[1] // kframes), frames_per_state=kframes)
                while not game.is_episode_finished():
                    frame = preprocess(game.get_state().screen_buffer)
                    frame = frame.reshape([1, 1, resolution[0], resolution[1] // kframes])
                    state = sb.get_state(frame)

                    best_action_index = get_best_action(state,preserve_state=False)

                    game.make_action(actions[best_action_index], frame_repeat)
                r = game.get_total_reward()
                test_scores.append(r)

            test_scores = np.array(test_scores)
            print("Results: mean: %.1f +/- %.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                  "max: %.1f" % test_scores.max())

            print("Saving the network weigths to:", model_savefile)
            model.save(model_savefile)

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

    game.close()
    print("======================================")
    print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        sb = StateBuilder((1, 1, resolution[0], resolution[1] // kframes), frames_per_state=kframes)
        while not game.is_episode_finished():
            frame = preprocess(game.get_state().screen_buffer)
            frame = frame.reshape([1, 1, resolution[0], resolution[1] // kframes])
            state = sb.get_state(frame)
            best_action_index = get_best_action(state,preserve_state=False)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)

