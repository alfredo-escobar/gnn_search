import tensorflow as tf


class SimilarityTensor():
    def __init__(self, embeddings, similarity_func):
        self.tensor = similarity_func(embeddings)
        self.realMin = tf.math.reduce_min(self.tensor)
        self.realMax = tf.math.reduce_max(self.tensor)
    
    def adjust_range(self, newMin, newMax, margin=0.0):

        newRange = newMax - newMin

        self.tensor = self.tensor - tf.math.reduce_min(self.tensor)
        # Current min value is 0

        self.tensor = self.tensor / tf.math.reduce_max(self.tensor)
        # Current max value is 1
        self.tensor = self.tensor * (newRange - 2*margin)
        # Current max value is newRange -2*margin

        self.tensor = self.tensor + newMin + margin
        # Current min value is newMin + margin
        # Current max value is newMax - margin
    
    def reset_range(self):
        self.adjust_range(self.realMin, self.realMax)


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