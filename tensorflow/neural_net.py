import tensorflow as tf
import numpy as np
training_epochs = 1500
n_neurons_in_h1 = 60
n_neurons_in_h2 = 60
learning_rate = 0.01

n_features = 784
n_classes = 10

tr_features = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1],
              [0, 0, 0]])
tr_labels = np.array([[0], [1], [1], [0], [0]])

ts_features = np.array([[0, 0, 1], [1, 0, 1]])
ts_labels = np.array([[0], [1]])

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = np.array(training_data)

test_data = np.array(training_data)
ts_features = np.array(list(map(lambda x: x.flatten(), test_data[:, 0])))
ts_labels = np.array(list(map(lambda x: x.flatten(), test_data[:, 1])))

# placeholdr tensors built to store features(in X) , labels(in Y) and dropout probability(in keep_prob)
X = tf.placeholder(tf.float32, [None, n_features], name='features')
Y = tf.placeholder(tf.float32, [None, n_classes], name='labels')
keep_prob=tf.placeholder(tf.float32,name='drop_prob')

W1 = tf.Variable(tf.truncated_normal([n_features, n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(n_features)), name='weights1')
b1 = tf.Variable(tf.truncated_normal([n_neurons_in_h1],mean=0, stddev=1 / np.sqrt(n_features)), name='biases1')
y1 = tf.nn.sigmoid((tf.matmul(X, W1)+b1), name='activationLayer1')

#network parameters(weights and biases) are set and initialized(Layer2)
W2 = tf.Variable(tf.random_normal([n_neurons_in_h1, n_neurons_in_h2],mean=0,stddev=1/np.sqrt(n_features)),name='weights2')
b2 = tf.Variable(tf.random_normal([n_neurons_in_h2],mean=0,stddev=1/np.sqrt(n_features)),name='biases2')
#activation function(sigmoid)
y2 = tf.nn.sigmoid((tf.matmul(y1,W2)+b2),name='activationLayer2')

#output layer weights and biasies
Wo = tf.Variable(tf.random_normal([n_neurons_in_h2, n_classes], mean=0, stddev=1/np.sqrt(n_features)), name='weightsOut')
bo = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=1/np.sqrt(n_features)), name='biasesOut')
#activation function(softmax)
a = tf.nn.sigmoid((tf.matmul(y2, Wo) + bo), name='activationOutputLayer')

#cost function
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(a),reduction_indices=[1]))
error = tf.square(Y - a)
#optimizer
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

#compare predicted value from network with the expected value/target
correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(Y, 1))
#accuracy determination
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

# initialization of all variables
initial = tf.global_variables_initializer()

#training_data = np.array(list(zip(tr_features, tr_labels)))
# creating a session
with tf.Session() as sess:
    sess.run(initial)
    # training loop over the number of epoches
    mini_batch_size=50
    for epoch in range(training_epochs):
        np.random.shuffle(training_data)
        mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
        for mini_batch in mini_batches:
            tr_features = np.array(list(map(lambda x: x.flatten(), mini_batch[:, 0])))
            tr_labels = np.array(list(map(lambda x: x.flatten(), mini_batch[:, 1])))
            # feeding training data/examples
            sess.run(train_step, feed_dict={X: tr_features, Y: tr_labels, keep_prob: 0.5})
        # feeding testing data to determine model accuracy
        y_pred = sess.run(a, feed_dict={X: ts_features, keep_prob: 1.0})
        y_true = sess.run(tf.constant(ts_labels))
        acc = sess.run(accuracy, feed_dict={X: ts_features, Y: ts_labels, keep_prob: 1.0})
        # write results to summary file
        # print accuracy for each epoch
        print('epoch', epoch, acc)
