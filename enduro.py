import gym
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D
from keras.optimizers import Adam
import numpy as np
import cv2
import random
import time
import gc
from collections import deque
import re
import os

class DQLAgent:

    def __init__(self, state_shape, action_size):
        self.action_size = action_size
        self.state_shape = state_shape
        self.epsilon = 0.2
        self.learning_rate = 0.1
        self.gamma = 0.90
        self.model = self._build_model()
        self.memory = deque(maxlen=2000) 

    def save_model(self, filename):
        self.model.save_weights(filename)

    def load_model(self, filename=None):
        try:
            if filename==None:
                file_list = os.listdir('save/')
                file_load = file_list[np.argmax([re.search(r'\s*([0-9]*)\.h5', file_entry).group(1) for file_entry in file_list])]
                self.model.load_weights('save/'+file_load)
            else:
                self.model.load_weights(filename)
        except:
            print("Nenhum modelo ajustado anteriormente.")

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(input_shape=self.state_shape, kernel_size=3, strides=1, filters=3, activation='relu'))
        model.add(MaxPool2D(strides=1, pool_size=2))
        model.add(Conv2D(kernel_size=3, strides=1, filters=1, activation='relu'))
        model.add(MaxPool2D(strides=1, pool_size=2))
        model.add(Conv2D(kernel_size=3, strides=1, filters=1, activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=512, activation='relu'))
        model.add(Dense(units=self.action_size, activation='linear'))
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        action_values = self.model.predict(state)
        return np.argmax(action_values[0])

    def replay(self):
        replay_batch = random.sample(self.memory, k=int(0.1*len(self.memory)))

        for state, action, reward, next_state, done in replay_batch:
            target = reward
            if not done:
                target = reward + self.gamma*(np.amax(self.model.predict(next_state)[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

if __name__ == '__main__':
    epochs = 100
    env = gym.make('Enduro-v0')
    state_shape = list(np.shape(env.reset()))
    state_shape[1] = state_shape[1]*2
    agent = DQLAgent(state_shape=state_shape, action_size=env.action_space.n)
    agent.load_model()

    for e in range(epochs):
        state = env.reset()
        state = np.concatenate((np.zeros(np.shape(state)), state), axis=1)
        state = np.expand_dims(state, axis=0)

        total_reward = 0
        while True:
            env.render()
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.concatenate((np.delete(state[0], slice(0,160), axis=1), next_state), axis=1)
            next_state = np.expand_dims(next_state, axis=0)
            agent.remember(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                print('play {}/{}, score = {}'.format(e, epochs, total_reward))
                break

        agent.replay()
    
        if e % 10 == 0:
            agent.save_model('save/modelo_enduro'+str(int(time.time()))+'.h5')

        gc.collect()
