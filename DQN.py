import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque



# Create a model class in Keras
class MyModel(keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__()

        # Initialize an input layer
        self.input_layer = keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []

        # Initialize the hidden layers
        for i in hidden_units:
            self.hidden_layers.append(keras.layers.Dense(i, activation='tanh'))

        # Initialize the output layer
        self.output_layer = keras.layers.Dense(num_actions, activation='linear')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)

        for layer in self.hidden_layers:
            z = layer(z)

        output = self.output_layer(z)
        return output

# Create a DQN class to handle the Q-networks
class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma,\
                        max_experiences, min_experiences, batch_size, lr):
        self.num_states = num_states
        self.num_actions = num_actions
        self.hidden_units = hidden_units
        self.gamma = gamma
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_size = batch_size
        self.lr = lr

        self.optimizer = tf.optimizers.Adam(self.lr)
        self.model = MyModel(num_states, hidden_units, num_actions)
        self.replay_buffer = {'s': [], 'a': [], 'r': [], 's_p': [], 'done': []}

    def predict(self, inputs):
        """ Accepts either a single state or a batch as the input,
            and runs a forward pass of self.model to return the
            model logits for actions.
        """
        return self.model(np.atleast_2d(inputs.astype('float32')))

    # @tf.function
    def train(self, TargetNet):
        if len(self.replay_buffer['s']) < self.min_experiences:
            return 0

        # print("here\n")
        # Sample a mini-batch
        ids = np.random.randint(low=0, high=len(self.replay_buffer['s']), \
                                                        size=self.batch_size)
        states = np.asarray([self.replay_buffer['s'][i] for i in ids])
        actions = np.asarray([self.replay_buffer['a'][i] for i in ids])
        rewards = np.asarray([self.replay_buffer['r'][i] for i in ids])
        next_states = np.asarray([self.replay_buffer['s_p'][i] for i in ids])
        dones = np.asarray([self.replay_buffer['done'][i] for i in ids])

        #next_value = np.max(TargetNet.predict(next_states), axis=1)
        next_value = np.asarray(TargetNet.predict(next_states))
        # print("Here\n")

        # y_t
        actual_value = np.where(dones, rewards, rewards + self.gamma * next_value)

        with tf.GradientTape() as tape:
            selected_q_values = tf.math.reduce_sum(self.model.predict(states) * \
                                tf.one_hot(actions, self.num_actions), axis=1)
            
            loss = tf.math.reduce_sum(tf.square(actual_value - selected_q_values))

        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        # return loss

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.uniform(0, 10000, self.num_actions)
        else:
#            return np.argmax(self.predict(np.atleast_2d(states))[0])
            return np.array(self.predict(np.atleast_2d(states))[0])

    def add_experience(self, exp):
        if len(self.replay_buffer['s']) >= self.max_experiences:
            for key in self.replay_buffer.keys():
                self.replay_buffer[key].pop(0)

        for key, value in exp.items():
            self.replay_buffer[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables

        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

def run(env, TrainNet, TargetNet, epsilon, copy_freq):
    rewards, iter = 0, 0
    done = False
    state = env.reset()

    while not done:
        action = TrainNet.get_action(state, epsilon) # have changed get_action, uncertain about uniform distribution 
        next_state, reward, done = env.step(action) 
        rewards += reward

        if done:
            reward = -200
            env.reset() 

        exp = {'s': state, 'a': action, 'r': reward, 's_p': next_state, 'done': done}
        TrainNet.add_experience(exp)
        TrainNet.train(TargetNet)

        iter += 1

        if iter % copy_freq == 0:
            TargetNet.copy_weights(TrainNet)

    return rewards