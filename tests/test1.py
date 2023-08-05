import numpy as np
import tensorflow as tf
from gnn_ssearch import train_new, loss_by_visual_text_contrast

# visual_embeddings = np.array([[ 1, 2, 3],
#                               [ 4, 5, 6],
#                               [ 7, 8, 9],
#                               [10,11,12]], dtype='float32')

# text_embeddings = np.array([[ 3, 4, 5],
#                             [ 6, 7, 8],
#                             [ 9,10,11],
#                             [12,13,14]], dtype='float32')

# new_features = train_new(visual_embeddings, text_embeddings, None, 20, 0.01)

similarity_text = np.array([[0.5, 0.2, 0.3],
                            [0.2, 0.7, 0.1],
                            [0.3, 0.1, 0.6]], dtype='float32')

similarity_visual = np.array([[0.6, 0.1, 0.3],
                              [0.1, 0.7, 0.2],
                              [0.3, 0.2, 0.5]], dtype='float32')

dissimilarity_text = tf.subtract(1.0, similarity_text)

visual_embeddings = np.array([[0.58682, 0.11314, 0.36261],
                              [0.11314, 0.58470, 0.19101],
                              [0.36261, 0.19101, 0.56315]], dtype='float32')

loss_by_visual_text_contrast(visual_embeddings, similarity_text, dissimilarity_text)