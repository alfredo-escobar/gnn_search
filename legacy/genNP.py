import numpy as np

def genNP(dataset):
    # Load the data from the .npy file
    a = np.load("./catalogues/{}/ssearch/visual_embeddings.npy".format(dataset))

    # Print the shape of the array to the console
    # print(a.shape)

    a.astype(np.float32).tofile("./catalogues/{}/ssearch/features.np".format(dataset))

    # Save the shape of the array to a new .npy file
    np.asarray(a.shape).astype(np.int32).tofile('./catalogues/{}/ssearch/features_shape.np'.format(dataset))