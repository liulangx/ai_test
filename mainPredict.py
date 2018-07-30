import random
import numpy as np
from AiGym import AiGymEnv

import tensorflow as tf
env = AiGymEnv()
inputSize = 7
outputSize = 8
x = tf.placeholder(dtype=tf.float32, shape=[None, inputSize], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[None, outputSize], name='y')

def add_layer(input_data, input_num, output_num, activation_function=None):
    w = tf.Variable(initial_value=tf.random_normal(shape=[input_num, output_num]))
    b = tf.Variable(initial_value=tf.random_normal(shape=[1, output_num]))
    output = tf.add(tf.matmul(input_data, w), b)
    if activation_function:
        output = activation_function(output)
    return output
def neural_network_model():
    # tf.sigmoid
    # tf.nn.leaky_relu
    hidden_layer1 = add_layer(x, inputSize, 128, activation_function=tf.sigmoid) # tf.sigmoid
    hidden_layer2 = add_layer(hidden_layer1, 128, 256, activation_function=tf.sigmoid)
    hidden_layer3 = add_layer(hidden_layer2, 256, 512, activation_function=tf.sigmoid)
    hidden_layer4 = add_layer(hidden_layer3, 512, 256, activation_function=tf.sigmoid)
    hidden_layer5 = add_layer(hidden_layer4, 256, 128, activation_function=tf.sigmoid)
    output_layer = add_layer(hidden_layer5, 128, outputSize, activation_function=tf.nn.softmax) # tf.nn.relu
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_layer)
    loss = tf.reduce_mean(loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2).minimize(loss)
    return output_layer, loss, optimizer

def predict(inputdata, sess, model):
    result = sess.run(model, feed_dict={x: inputdata})
    action = np.argmax(result[0])
    return action

scores = []
choices = []
with tf.Session() as sess:
    output, loss, optimizer = neural_network_model()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, './model.ckpt')

    for each_game in range(100):
        score = 0
        game_memory = []
        prev_obs = []
        prev_obs = env.reset()

        for i in range(1000):
            env.render()
            # if len(prev_obs) == 0:
            #     action = random.randrange(0, 8)
            # else:
            action = predict(prev_obs.reshape(-1, inputSize), sess, output)
            choices.append(action)

            new_observation, reward, done, inDrugArea, intoDrugArea, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward
            print(i, ": ", new_observation, " ", done)
            if done:
                print("i:{}".format(i))
                break
        scores.append(score)
    print('Scores: ', scores)
    print('Average Score: ', sum(scores) / len(scores))
    print('choice 1: {}, choice 0: {}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))
    print('choice 3: {}, choice 2: {}'.format(choices.count(3) / len(choices), choices.count(2) / len(choices)))
    print('choice 5: {}, choice 4: {}'.format(choices.count(5) / len(choices), choices.count(4) / len(choices)))
    print('choice 7: {}, choice 6: {}'.format(choices.count(7) / len(choices), choices.count(6) / len(choices)))