import tensorflow as tf

a = tf.Variable([[ 1, 2, 3, 4],
                 [ 5, 6, 7, 8],
                 [ 9,10,11,12]])

b = tf.Variable([[ 1, 2, 2, 1],
                 [ 1, 2, 1, 1],
                 [ 2, 1, 1, 2]])

a = a*b

print(a.numpy())