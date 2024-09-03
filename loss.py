import tensorflow as tf


def loss_by_visual_text_contrast(similarity_visual, similarity_text, batch_indexes, tau=1):

    similarity_idx = tf.transpose(tf.convert_to_tensor(batch_indexes))

    subsimilarity_text = tf.gather_nd(similarity_text, similarity_idx)
    subsimilarity_visual = tf.gather_nd(similarity_visual, similarity_idx)

    dissimilarity_text = tf.subtract(1.0, subsimilarity_text)
    dissimilarity_visual = tf.subtract(1.0, subsimilarity_visual)

    similarity_weighted = tf.divide(subsimilarity_text, subsimilarity_visual)
    dissimilarity_weighted = tf.divide(dissimilarity_text, dissimilarity_visual)

    similars_addend = tf.multiply(subsimilarity_text, tf.math.log(similarity_weighted))
    dissimilars_addend = tf.multiply(dissimilarity_text, tf.math.log(dissimilarity_weighted))

    similars_addend = tf.maximum(similars_addend, 0)
    dissimilars_addend = tf.maximum(dissimilars_addend, 0)

    return tf.reduce_mean(tf.add(similars_addend, dissimilars_addend))


def get_prob(similarity, similarity_idx, alpha, margin):

    # We select only the similarity values from the random batch
    prob = tf.gather_nd(similarity, similarity_idx)

    # We adjust the range in order to obtain values from exp func
    prob -= tf.reduce_min(similarity)
    prob /= (tf.reduce_max(similarity) - tf.reduce_min(similarity))
    prob *= 2 * alpha
    prob -= alpha
    prob = tf.exp(prob)

    # Margin adjustment
    prob /= tf.exp(alpha)
    prob *= (1.0 - 2*margin)
    prob += margin

    return prob


def loss_unet(similarity_visual, similarity_text, batch_indexes, alpha = 10.0, margin = 0.001, ratio = 0.8):

    similarity_idx = tf.transpose(tf.convert_to_tensor(batch_indexes))

    prob_gt = get_prob(similarity_text, similarity_idx, alpha, margin)
    #prob_gt = tf.gather_nd(similarity_text, similarity_idx)
    #prob_pred = tf.gather_nd(similarity_visual, similarity_idx)
    prob_pred = get_prob(similarity_visual, similarity_idx, alpha, margin)

    sims_and_probs = {
        "node_1" : batch_indexes[0],
        "node_2" : batch_indexes[1],
        "blank1" : [],
        "sim_text" : tf.gather_nd(similarity_text, similarity_idx).numpy(),
        "prob_gt" : prob_gt.numpy(),
        "blank2" : [],
        "sim_visual" : tf.gather_nd(similarity_visual, similarity_idx).numpy(),
        "prob_pred" : prob_pred.numpy()
    }

    similars_addend = tf.maximum((prob_gt * tf.math.log(prob_gt / prob_pred)), 0)
    dissimilars_addend = tf.maximum(((1 - prob_gt) * tf.math.log((1 - prob_gt) / (1 - prob_pred))), 0)

    similars_addend = tf.multiply(similars_addend, ratio)
    dissimilars_addend = tf.multiply(dissimilars_addend, 1-ratio)

    return tf.reduce_mean(tf.add(similars_addend, dissimilars_addend)), sims_and_probs