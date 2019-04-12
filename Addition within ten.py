import tensorflow as tf
import numpy as np
input = 2
label = 1
hidden1 = 2
x = tf.placeholder(tf.float32,[None,input])
y = tf.placeholder(tf.float32,[None,label])
weights = {
    'h1':tf.Variable(tf.truncated_normal([input,hidden1],stddev=0.1)),
    'h2':tf.Variable(tf.truncated_normal([hidden1,label],stddev=0.1))
}
biases = {
    'h1':tf.Variable(tf.zeros([hidden1])),
  'h2':tf.Variable(tf.zeros([label]))
}
layer1 = tf.nn.relu(tf.add(tf.matmul(x,weights['h1']),biases['h1']))
y_pred = tf.add(tf.matmul(layer1,weights['h2']),biases['h2'])
loss= tf.reduce_mean((y-y_pred)**2)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
X = [[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],[1,9],
     [2,1],[2,2],[2,3],[2,4],[2,5],[2,6],[2,7],[2,8],
     [3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],
     [4,1],[4,2],[4,3],[4,4],[4,5],[4,6],
     [5,1],[5,2],[5,3],[5,4],[5,5],
     [6,1],[6,2],[6,3],[6,4],
     [7,1],[7,2],[7,3],
     [8,1],[8,2],
     [9,1]]
Y = [[2],[3],[4],[5],[6],[7],[8],[9],[10]
    ,[3],[4],[5],[6],[7],[8],[9],[10],
     [4],[5],[6],[7],[8],[9],[10],
     [5],[6],[7],[8],[9],[10],
     [6],[7],[8],[9],[10],
     [7],[8],[9],[10],
     [8],[9],[10],
     [9],[10],
     [10]]
X = np.array(X).astype('float32')
Y = np.array(Y).astype('int16')
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(10000):
    sess.run(train_step,feed_dict={x:X,y:Y})
print(sess.run(y_pred,feed_dict={x:X}))