import numpy as np
import tensorflow as tf


class Similarity:
    def __init__(self, features, type = 'cos'):

        if type == 'cos':
            similarity_symmetrical = self.get_cos_similarity_tensor(features)
        elif type == 'cos_softmax':
            similarity_symmetrical = self.get_cos_softmax_similarity_tensor(features)
        elif type == 'sqrt':
            similarity_symmetrical = self.get_sqrt_similarity_tensor(features)
        elif type == 'sqrt_normmin':
            similarity_symmetrical = self.get_sqrt_normmin_similarity_tensor(features)
        
        self.size = features.shape[0]

        similarity_idx = tf.convert_to_tensor(np.triu_indices(self.size))

        self.similarity = tf.gather_nd(similarity_symmetrical, tf.transpose(similarity_idx))









    def get_k_random_pairs(self, k = 100):

        # 2D indexes for the similarity matrix:
        similarity_idx = np.triu_indices(self.size)

        # 1D array with similarity scores of upper half of matrix
        similarity_top = similarity[similarity_idx]

        # plot_histogram_2(similarity_top)

        # 1D indexes for the 2D indexes for the similarity matrix:
        indexes = np.arange(similarity_idx[0].shape[0])
        size = int(k/2)

        probs_sim    = similarity_top / np.sum(similarity_top)
        probs_dissim = (1.0 - similarity_top) / np.sum(1.0 - similarity_top)

        # k random 1D indexes for the 2D indexes for the similarity matrix:
        random_indexes_similar    = np.random.choice(indexes, size, replace=False, p=probs_sim)
        random_indexes_dissimilar = np.random.choice(indexes, size, replace=False, p=probs_dissim)
        # probgt = 
        batch_indexes = np.concatenate((random_indexes_similar, random_indexes_dissimilar))

        return [similarity_idx[0][batch_indexes], similarity_idx[1][batch_indexes]]
        #prob_gt = similarity[batch_indexes] # 

        # plot_histogram(similarity_top, random_indexes_similar, random_indexes_dissimilar)
        
        # batch_mask = np.zeros((similarity.shape[0], similarity.shape[0]), dtype='float32')
        
        # batch_mask[similarity_idx[0][random_indexes_similar],    similarity_idx[1][random_indexes_similar]] = 1.0
        # batch_mask[similarity_idx[0][random_indexes_dissimilar], similarity_idx[1][random_indexes_dissimilar]] = 1.0

        # return tf.convert_to_tensor(batch_mask)


    def get_euclidean_distances(embeddings):

        print("---Computing euclidean distances...---")

        size = embeddings.shape[0]

        distances_list = []

        for idx in range(size):

            diff_per_dim = tf.subtract(embeddings[idx], embeddings)

            dists_from_this = tf.norm(diff_per_dim, ord='euclidean', axis=1)

            distances_list.append(dists_from_this)

        distances = tf.stack(distances_list)

        return distances


    def make_symmetrical(similarity):
        similarity_transposed = tf.transpose(similarity)
        multiplied_sim_simt = tf.multiply(similarity, similarity_transposed)

        similarity = tf.add(similarity, similarity_transposed)
        similarity = tf.subtract(similarity, multiplied_sim_simt)

        return similarity


    def get_cos_similarity_tensor(embeddings):
        normalized_embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        similarity = tf.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True)
        return similarity


    def get_cos_softmax_similarity_tensor(embeddings):
        # We get simiarity by cosine distance
        similarity = get_cos_similarity_tensor(embeddings)

        similarity = tf.nn.softmax(similarity)
        # Up until this point, all values in each row add up to 1
        # The similarity tensor became non symmetrical

        similarity = make_symmetrical(similarity)
        # Now, the similarity tensor is symmetrical again        

        return similarity


    def get_sqrt_similarity_tensor(embeddings):
        distances = get_euclidean_distances(embeddings)

        similarity_denom = tf.exp(-distances)
        similarity = tf.divide(1.0, similarity_denom)

        similarity = tf.nn.softmax(similarity)
        similarity = make_symmetrical(similarity)

        return similarity


    def get_sqrt_normmin_similarity_tensor(embeddings):
        distances = get_euclidean_distances(embeddings)
        min_distances = tf.expand_dims(tf.reduce_min(distances, axis=1), axis=1)

        similarity_denom = tf.exp(-(distances-min_distances))
        similarity = tf.divide(1.0, similarity_denom)

        similarity = tf.nn.softmax(similarity)
        similarity = make_symmetrical(similarity)

        return similarity