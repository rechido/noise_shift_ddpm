"""
prdc 
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import numpy as np
import sklearn.metrics

__all__ = ['compute_prdc']

debug=False

def compute_pairwise_distance(data_x, data_y=None, distance_metric='euclidean'):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x

    if distance_metric.lower() in ['euclidean', 'l2', 'l2-norm']:
        dists = sklearn.metrics.pairwise_distances(data_x, data_y, metric='euclidean', n_jobs=8)

    elif distance_metric.lower() in ['cosine_similarity', 'cos', 'cosine']:
        dists = sklearn.metrics.pairwise.cosine_distances(data_x, data_y) # 1.0 minus Cosine Similarity #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_distances.html

    else:
        print(f'error: not implemented distance metric {distance_metric}')
        exit(-1)

    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices     = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values  = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k, distance_metric):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features, distance_metric=distance_metric)
    radii     = get_kth_value(distances, k=nearest_k + 1, axis=-1)

    if debug:
        print('compute_nearest_neighbour_distances: distances.shape',distances.shape) #(10000, 10000)
        print('compute_nearest_neighbour_distances: radii.shape',    radii.shape)     #(10000)
 
    return radii


def compute_prdc(real_features, fake_features, nearest_k, distance_metric):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """

    print('Num real: {} Num fake: {}'
          .format(real_features.shape[0], fake_features.shape[0]))

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances( real_features, nearest_k, distance_metric=distance_metric )
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances( fake_features, nearest_k, distance_metric=distance_metric )
    distance_real_fake               = compute_pairwise_distance( real_features, fake_features, distance_metric=distance_metric )

    if debug:
        real_nearest_neighbour_distances = compute_nearest_neighbour_distances( real_features, nearest_k, distance_metric=distance_metric )
        fake_nearest_neighbour_distances = compute_nearest_neighbour_distances( fake_features, nearest_k, distance_metric=distance_metric )
        distance_real_fake               = compute_pairwise_distance( real_features, fake_features, distance_metric=distance_metric )
        print('real_nearest_neighbour_distances.shape:', real_nearest_neighbour_distances.shape) # [real][real]
        print('fake_nearest_neighbour_distances.shape:', fake_nearest_neighbour_distances.shape) # [fake][fake]
        print('distance_real_fake.shape', distance_real_fake.shape)                              # [real][fake]


    precision = ( distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)).any(axis=0).mean()
    recall    = ( distance_real_fake < np.expand_dims(fake_nearest_neighbour_distances, axis=0)).any(axis=1).mean()
    density   = (1. / float(nearest_k)) * ( distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1) ).sum(axis=0).mean()
    coverage  = ( distance_real_fake.min(axis=1) < real_nearest_neighbour_distances ).mean()

    fake_real_distance     = distance_real_fake.min(axis=0) # the closest real from each fake, 
    fake_real_min_distance = fake_real_distance.mean()

    return dict(precision=precision, recall=recall, density=density, coverage=coverage, fake_real_min_distance=fake_real_min_distance)




# Added by Da Eun Lee, AI department, Chung-Ang University.
# A criterion for replicate detection, following "Diffusion Probabilistic Models Generalize when They Fail to Memorize".
# Different with the original work, we compute it in the feature space.
# weight [3] is recommended in the original paper.
def compute_replicate(real_features, fake_features, distance_metric='euclidean', weight_set=[2,3,4]):

    #distance_metric    = 'euclidean' # should be L2 norm 
    distance_real_fake = compute_pairwise_distance(real_features, fake_features, distance_metric=distance_metric)

    # Find 1st and 2nd smallest values
    # https://stackoverflow.com/questions/44002239/how-to-get-the-two-smallest-values-from-a-numpy-array
    first_smallest,second_smallest = np.partition(distance_real_fake, 1, axis=0)[0:2] # top only (k+1) in incremental order

    sorted_indices_fake = np.argsort(distance_real_fake, axis=0)
    first_smallest_real_indices = sorted_indices_fake[0]
    second_smallest_real_indices = sorted_indices_fake[1]
    
    sorted_indices_real = np.argsort(distance_real_fake, axis=1)
    first_smallest_fake_indices = sorted_indices_real[:,0]
    second_smallest_fake_indices = sorted_indices_real[:,1]

    dictionary_replicate={}
    for weight in weight_set:
        replicate = ( weight * first_smallest < second_smallest ).mean()
        dictionary_replicate[f"replicate({weight})"] = replicate

    return dictionary_replicate, first_smallest_real_indices, second_smallest_real_indices, first_smallest_fake_indices, second_smallest_fake_indices


# A criterion for replicate detection, utilizing absolute threshold value
def compute_replicate_abs(real_features, fake_features, thresholds, distance_metric='euclidean'):
    
    distance_real_fake = compute_pairwise_distance(real_features, fake_features, distance_metric=distance_metric)
    
    first_smallest = np.partition(distance_real_fake, 1, axis=0)[0:1]
    
    dictionary_replicate={}
    for threshold in thresholds:
        replicate = (first_smallest < threshold).mean()
        dictionary_replicate[f"replicate({threshold})"] = replicate
    
    return dictionary_replicate, first_smallest
