import numpy as np

def get_k_random_pairs(similarity, k = 100):

    # 2D indexes for the similarity matrix:
    similarity_idx = np.triu_indices(similarity.shape[0])

    # 1D array with similarity scores of upper half of matrix
    similarity_top = similarity[similarity_idx]

    # 1D indexes for the 2D indexes for the similarity matrix:
    indexes = np.arange(similarity_idx[0].shape[0])
    size = int(k/2)

    probs_sim    = similarity_top / np.sum(similarity_top)
    probs_dissim = (1.0 - similarity_top) / np.sum(1.0 - similarity_top)

    # k random 1D indexes for the 2D indexes for the similarity matrix:
    random_indexes_similar    = np.random.choice(indexes, size, replace=False, p=probs_sim)
    random_indexes_dissimilar = np.random.choice(indexes, size, replace=False, p=probs_dissim)

    batch_mask = np.zeros((similarity.shape[0], similarity.shape[0]), dtype='float32')
    
    batch_mask[similarity_idx[0][random_indexes_similar],    similarity_idx[1][random_indexes_similar]] = 1.0
    batch_mask[similarity_idx[0][random_indexes_dissimilar], similarity_idx[1][random_indexes_dissimilar]] = 1.0

    return batch_mask

a = np.array([[ 1, 2, 3, 4],
              [ 5, 6, 7, 8],
              [ 9,10,11,12],
              [13,14,15,16]])

b = np.triu_indices(a.shape[0],1)

print("breakpoint")