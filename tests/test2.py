import tensorflow as tf

a = tf.Variable([[5, 8, 1],
                 [1, 3, 4],
                 [6, 5, 2],
                 [7, 9, 1]], dtype=tf.float32)

b = tf.Variable([20,20,20], dtype=tf.float32)

c = tf.norm(a, ord='euclidean', axis=1)

indexes_x = tf.Variable([[0,1],[0,1],[0,1]], dtype=tf.int32)

d = a[indexes_x]

print("stop")