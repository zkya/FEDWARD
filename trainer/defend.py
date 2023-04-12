import numpy as np
import sklearn.metrics.pairwise as smp
import matplotlib.pyplot as plt
import scipy.stats


# Simple element-wise median
def median(g_deltas):

    return np.median(g_deltas, axis=0)

# Beta is the proportion to trim from the top and bottom.
def trimmed_mean(g_deltas, beta = 0.1):

    return scipy.stats.trim_mean(g_deltas, beta, axis=0)


# Returns the index of the row that should be used in Krum
def krum(deltas, n_client, clip):

    # assume deltas is an array of size group * d
    n = len(deltas)
    scores = get_krum_scores(deltas, n_client - clip)
    good_idx = np.argpartition(scores, n_client - clip)[:(n_client - clip)]

    return np.mean(deltas[good_idx], axis=0)


def get_krum_scores(X, groupsize):

    krum_scores = np.zeros(len(X))

    # Calculate distances
    distances = np.sum(X**2, axis=1)[:, None] + np.sum(
        X**2, axis=1)[None] - 2 * np.dot(X, X.T)

    for i in range(len(X)):
        krum_scores[i] = np.sum(np.sort(distances[i])[1:(groupsize - 1)])

    return krum_scores

# Simple element-wise mean
def average(g_deltas):
    return np.mean(g_deltas, axis=0)


'''
Aggregates history of gradient directions
'''
def foolsgold(this_delta, summed_deltas, sig_features_idx, iter, model, topk_prop=0, importance=False, importanceHard=False, clip=0):

    # Take all the features of sig_features_idx for each clients
    sd = summed_deltas.copy()
    sig_filtered_deltas = np.take(sd, sig_features_idx, axis=1)

    if importance or importanceHard:
        if importance:
            # smooth version of importance features
            importantFeatures = importanceFeatureMapLocal(model, topk_prop)
        if importanceHard:
            # hard version of important features
            importantFeatures = importanceFeatureHard(model, topk_prop)
        for i in range(n):
            sig_filtered_deltas[i] = np.multiply(sig_filtered_deltas[i], importantFeatures)

    cs = smp.cosine_similarity(sig_filtered_deltas) - np.eye(n)
     
    # Pardoning: reweight by the max value seen
    maxcs = np.max(cs, axis=1) + epsilon
     
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

    wv = 1 - (np.max(cs, axis=1)) 
     
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99
    
    # Logit function
    wv = (np.log((wv / (1 - wv)) + epsilon) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0
    
    # if iter % 10 == 0 and iter != 0:
    #     print maxcs
    #     print wv

    if clip != 0:

        # Augment onto krum
        scores = get_krum_scores(this_delta, n - clip)
        bad_idx = np.argpartition(scores, n - clip)[(n - clip):n]

        # Filter out the highest krum scores
        wv[bad_idx] = 0 
    # Apply the weight vector on this delta
    delta = np.reshape(this_delta, (n, d)) 
    return np.dot(delta.T, wv)

