import matplotlib.pyplot as plt

def plot_mAP(historical_mAP):
    if len(historical_mAP["mAPs"]) > 0:
        fig, ax = plt.subplots()

        ax.plot(historical_mAP["iters"], historical_mAP["mAPs"], 'o-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('mAP')
        ax.set_title('mAP in training')

        #plt.xticks(color='w')
        plt.show()


def plot_loss(historical_loss):
    fig, ax = plt.subplots()

    ax.plot(historical_loss["iters"], historical_loss["losses"], 'o-')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Loss in training')

    plt.show()


def plot_probs(similarity_1D, probs):
    fig, ax = plt.subplots()
    ax.scatter(similarity_1D, probs)
    ax.set_xlabel("Similarity score")
    ax.set_ylabel("Probability of being chosen")
    plt.show()


def plot_histogram_randoms(similarity_1D, random_indexes_similar, random_indexes_dissimilar):
    num_bins = 50

    fig, ax = plt.subplots()

    x_multi = [similarity_1D[random_indexes_similar],
               similarity_1D[random_indexes_dissimilar]]
    
    colors = ['green', 'red']
    labels = ['Randomly chosen similar pairs', 'Randomly chosen dissimilar pairs']

    ax.hist(x_multi, num_bins, histtype='bar', color=colors, label=labels)

    ax.set_xlabel('Similarity score')
    ax.set_ylabel('# of node pairs')
    ax.set_title('Amount of node pairs chosen for random batches, by similarity score (in bins)')
    plt.legend(loc="upper right")

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()


def plot_histogram_sim(similarity_1D):
    num_bins = 50

    fig, ax = plt.subplots()

    x_multi = [similarity_1D]
    
    colors = ['blue']

    ax.hist(x_multi, num_bins, histtype='bar', color=colors)

    ax.set_xlabel('Similarity score')
    ax.set_ylabel('# of node pairs')
    ax.set_title('Amount of node pairs by similarity score (in bins)')

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()

def plot_prob_gt(all_sims_and_probs):
    num_bins = 50
    fig, axs = plt.subplots(len(all_sims_and_probs), 1)

    for idx, key in enumerate(all_sims_and_probs):
        axs[idx].hist(all_sims_and_probs[key]['prob_gt'], num_bins, histtype='bar')
        axs[idx].set_xlabel('prob_gt score')
        axs[idx].set_ylabel('# of node pairs')
        axs[idx].set_title('Amount of node pairs by prob_gt (in bins) on ' + key)

    fig.tight_layout()
    plt.show()