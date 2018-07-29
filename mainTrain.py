# -*-coding: utf-8 -*-
# import gym
#from aiGym import AiGymEnv
from statistics import median, mean
from collections import Counter
import os
import random
import numpy as np
import tensorflow as tf
from AiGym import AiGymEnv


LR = 1e-3

# env = gym.make("CartPole-v0")
env = AiGymEnv()
env.reset()

goal_steps = 1000
score_requirement = 24
initial_games = 50000000
roundDisp = 10000
batch_size = 512
train_times = 20
gpu_options = tf.GPUOptions(allow_growth=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

inputSize = 7
outputSize = 8
x = tf.placeholder(dtype=tf.float32, shape=[None, inputSize], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[None, outputSize], name='y')

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for round in range(initial_games):
        if round % roundDisp == 0:
            print("round: ", round)
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0, 8)
            observation, reward, done, inDrugArea, intoDrugArea, info = env.step(action)
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                # print("round {}:".format(round), score)
                if inDrugArea:
                    score -= 30
                if intoDrugArea:
                    score -= 10
                # print("round {}:".format(round), score)
                break
        if score >= score_requirement * observation[0] / 12:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 7:
                    output = [0, 0, 0, 0, 0, 0, 0, 1]
                elif data[1] == 6:
                    output = [0, 0, 0, 0, 0, 0, 1, 0]
                elif data[1] == 5:
                    output = [0, 0, 0, 0, 0, 1, 0, 0]
                elif data[1] == 4:
                    output = [0, 0, 0, 0, 1, 0, 0, 0]
                elif data[1] == 3:
                    output = [0, 0, 0, 1, 0, 0, 0, 0]
                elif data[1] == 2:
                    output = [0, 0, 1, 0, 0, 0, 0, 1]
                elif data[1] == 1:
                    output = [0, 1, 0, 0, 0, 0, 0, 0]
                elif data[1] == 0:
                    output = [1, 0, 0, 0, 0, 0, 0, 0]
                training_data.append([data[0], output])
        env.reset()
        scores.append(score)
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)
    print('Average accepted score:', mean(accepted_scores))
    print('Median score for accepted scores:', median(accepted_scores))
    print(Counter(accepted_scores))
    print(len(accepted_scores))

    return training_data
def add_layer(input_data, input_num, output_num, activation_function=None):
    w = tf.Variable(initial_value=tf.random_normal(shape=[input_num, output_num]))
    b = tf.Variable(initial_value=tf.random_normal(shape=[1, output_num]))
    output = tf.add(tf.matmul(input_data, w), b)
    if activation_function:
        output = activation_function(output)
    return output
def neural_network_model(input_size):
    # tf.sigmoid
    # tf.nn.leaky_relu
    hidden_layer1 = add_layer(x, input_size, 128, activation_function=tf.sigmoid) # tf.sigmoid
    hidden_layer2 = add_layer(hidden_layer1, 128, 256, activation_function=tf.sigmoid)
    hidden_layer3 = add_layer(hidden_layer2, 256, 512, activation_function=tf.sigmoid)
    hidden_layer4 = add_layer(hidden_layer3, 512, 256, activation_function=tf.sigmoid)
    hidden_layer5 = add_layer(hidden_layer4, 256, 128, activation_function=tf.sigmoid)
    output_layer = add_layer(hidden_layer5, 128, 2, activation_function=tf.nn.softmax) # tf.nn.relu
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_layer)
    loss = tf.reduce_mean(loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2).minimize(loss)
    return output_layer, loss, optimizer

def train_model(training_data, model = False):
    x_data = np.array([i[0] for i in training_data]).reshape(-1, inputSize)
    y_data = np.array([i[1] for i in training_data]).reshape(-1, outputSize)
    if not model:
        output, loss, optimizer = neural_network_model(inputSize)
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        if not os.path.exists('checkpoint'):
            for i in range(train_times):
                lenDataset = x_data.shape[0]
                each_cost = 0;
                for j in range(lenDataset // batch_size + 1):
                    if (j + 1) * batch_size < len(x_data):
                        x_data_split = x_data[j * batch_size: (j + 1) * batch_size, :]
                        y_data_split = y_data[j * batch_size: (j + 1) * batch_size, :]
                    else:
                        x_data_split = x_data[j * batch_size: len(x_data),:]
                        y_data_split = y_data[j * batch_size: len(x_data), :]
                    cost, _ = sess.run([loss, optimizer], feed_dict={x: x_data_split, y: y_data_split})
                    each_cost += cost
                print('Epoch', i, ":", each_cost)
            saver.save(sess, './model.ckpt')
        else:
            saver.restore(sess, './model.ckpt')
training_data = initial_population()
train_model(training_data)
