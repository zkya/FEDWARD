import sys

import hdbscan
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN, AgglomerativeClustering, OPTICS, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

np.set_printoptions(threshold=sys.maxsize)


def auto_optics(weight_list, args):
    weight_list = np.array(weight_list, dtype='float64')
    distance_matrix = cdist(weight_list, weight_list, metric='euclidean')
    size = len(weight_list)

    # svd = TruncatedSVD(n_components=(size) // 2 + 1, random_state=3)
    # decomp_updates = svd.fit_transform(distance_matrix.T) # shape=(n_params, n_groups)

    # n_components = decomp_updates.shape[-1]
    # print("n_components", n_components)

    # edulic = np.sqrt(edulic)
    # print("edumatirx", distance_matrix)

    Eps = np.median(np.sort(distance_matrix, axis=1)[:, (size)// 2])

    clusterer = OPTICS(min_samples = (size)// 2, max_eps=np.inf, 
    eps = Eps, cluster_method = 'dbscan',metric = 'precomputed').fit(distance_matrix)

    a = {}
    for i in range(max(clusterer.labels_) + 1):
        a[i] = clusterer.labels_.tolist().count(i) 
    admitted_label = 0 
    for i in a:
        if a[i] == max(a.values()):
            admitted_label = i
            break
    admitted_index = [] 

    for i in range(len(clusterer.labels_)):  # len(clusterer.labels_) = len(local_choose)
        if clusterer.labels_[i] == admitted_label:
            admitted_index.append(i)
    return admitted_index


def model_filtering_layer(weight_list, args):
    weight_list = np.array(weight_list, dtype='float64')
    if args["sign"]:
        distance_matrix = cdist(weight_list, weight_list, metric='hamming')
    else:
        distance_matrix = cosine_distances(weight_list)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=len(weight_list) // 2 + 1,
                                gen_min_span_tree=True,
                                metric='precomputed', allow_single_cluster=True, alpha=0.9)
    clusterer.fit(distance_matrix)
    a = {}
    for i in range(max(clusterer.labels_) + 1):
        a[i] = clusterer.labels_.tolist().count(i) 
    admitted_label = 0 
    for i in a:
        if a[i] == max(a.values()):
            admitted_label = i
            break
    admitted_index = []  

    for i in range(len(clusterer.labels_)):
        if clusterer.labels_[i] == admitted_label:
            admitted_index.append(i)

    return admitted_index


def model_filtering_layer_fedcc(weight_list, args):

    weight_list = np.array(weight_list, dtype='float64')
    clusterer = KMeans(2).fit(weight_list)
    a = {}
    for i in range(max(clusterer.labels_) + 1):
        a[i] = clusterer.labels_.tolist().count(i)  
    admitted_label = 0 
    for i in a:
        if a[i] == max(a.values()):
            admitted_label = i
            break
    admitted_index = [] 

    for i in range(len(clusterer.labels_)):  
        if clusterer.labels_[i] == admitted_label:
            admitted_index.append(i)
    return admitted_index



def adaptive_clipping(global_weight, weight_list, admitted_index):
    E = euclidean_distances([global_weight], weight_list)[0]
    St = np.median(E)
    for i in admitted_index:
        weight_list[i] = global_weight + (weight_list[i] - global_weight) * min(1, St / E[i])
    return weight_list, St


def get_penultimate_layer(args, weights):
    if args['dataset'] == 'MNIST' or args['dataset'] == 'FASHION':
        plr_name = ['fc2.weight', 'fc2.bias']
    else:
        plr_name = ['classifier.6.weight', 'classifier.6.bias']
    ret = []
    with torch.no_grad():
        for name in plr_name:
            tt = weights.state_dict()[name]
            tt = torch.where(torch.isnan(tt), torch.full_like(tt, 0), tt)
            ret = ret + tt.flatten().cpu().tolist()
    ret = np.array(ret)
    return ret
    

def AMGRAD(list_of_weights, aim_net, single=False):
    ret = []
    for i in list_of_weights:
        reshaped = np.array([])
        start = 0
        for p in aim_net.parameters():
            end = start + len(p.view(-1))
            tmp = i[start:end]
            signn = np.sign(tmp)
            maxn = np.linalg.norm(tmp, ord = np.inf)
            tmp = signn * tmp
            am = np.round(tmp / maxn) * maxn
            am = signn * am
            reshaped = np.hstack((reshaped, am))
            start = end
        ret.append(reshaped)
    return ret



def adaptive_noising(net, alpha):
    total_length = 0
    for p in net.parameters():
        total_length += p.view(-1).shape[0]
    noise = torch.normal(0.0, alpha, [total_length]).to(next(net.parameters()).device)
    start = 0
    i = 0
    for p in net.parameters():
        length = p.view(-1).shape[0]
        end = start + length
        p.data += noise[start:end].view_as(p)
        start = end
