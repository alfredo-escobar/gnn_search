import numpy as np
import tensorflow as tf

# Example 2D tensor 'a'
a = tf.constant([[ 1, 2, 3, 4],
                 [ 5, 6, 7, 8],
                 [ 9,10,11,12],
                 [13,14,15,16]])

similarity_idx = tf.convert_to_tensor(np.triu_indices(4))

# Define the indices for 'b'
indices = tf.constant([[0, 1],
                       [1, 2],
                       [2, 0]])

# Use tf.gather_nd() to gather elements from 'a' based on indices
#b = tf.gather_nd(a, tf.transpose(similarity_idx))
b = tf.gather_nd(a, indices)

# Print the result
print(b.numpy())  # Output: [2 6 7]
