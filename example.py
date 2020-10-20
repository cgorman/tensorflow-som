import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_som import SelfOrganizingMap
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import logging
from scipy.spatial import distance_matrix

'''
An example usage of the TensorFlow SOM. Loads a data set, trains a SOM, and displays the u-matrix.
'''


def get_umatrix(input_vects, weights, m, n):
    """ Generates an n x m u-matrix of the SOM's weights and bmu indices of all the input data points

    Used to visualize higher-dimensional data. Shows the average distance between a SOM unit and its neighbors.
    When displayed, areas of a darker color separated by lighter colors correspond to clusters of units which
    encode similar information.
    :param weights: SOM weight matrix, `ndarray`
    :param m: Rows of neurons
    :param n: Columns of neurons
    :return: m x n u-matrix `ndarray` 
    :return: input_size x 1 bmu indices 'ndarray'
    """
    umatrix = np.zeros((m * n, 1))
    # Get the location of the neurons on the map to figure out their neighbors. I know I already have this in the
    # SOM code but I put it here too to make it easier to follow.
    neuron_locs = list()
    for i in range(m):
        for j in range(n):
            neuron_locs.append(np.array([i, j]))
    # Get the map distance between each neuron (i.e. not the weight distance).
    neuron_distmat = distance_matrix(neuron_locs, neuron_locs)

    for i in range(m * n):
        # Get the indices of the units which neighbor i
        neighbor_idxs = neuron_distmat[i] <= 1  # Change this to `< 2` if you want to include diagonal neighbors
        # Get the weights of those units
        neighbor_weights = weights[neighbor_idxs]
        # Get the average distance between unit i and all of its neighbors
        # Expand dims to broadcast to each of the neighbors
        umatrix[i] = distance_matrix(np.expand_dims(weights[i], 0), neighbor_weights).mean()

    bmu_indices = []
    for vect in input_vects:
        min_index = min([i for i in range(len(list(weights)))],
                        key=lambda x: np.linalg.norm(vect-
                                                     list(weights)[x]))
        bmu_indices.append(neuron_locs[min_index])
        
    return umatrix, bmu_indices


def get_umatrix_optimized(input_vects, weights, m, n):
    """ Generates an n x m u-matrix of the SOM's weights and bmu indices of all the input data points

    Used to visualize higher-dimensional data. Shows the average distance between a SOM unit and its neighbors.
    When displayed, areas of a darker color separated by lighter colors correspond to clusters of units which
    encode similar information.
    :param weights: SOM weight matrix, `ndarray`
    :param m: Rows of neurons
    :param n: Columns of neurons
    :return: m x n u-matrix `ndarray` 
    :return: input_size x 1 bmu indices 'ndarray'
    """
    umatrix = np.zeros((m * n, 1))
    # Get the location of the neurons on the map to figure out their neighbors. I know I already have this in the
    # SOM code but I put it here too to make it easier to follow.
    neuron_locs = list()
    for i in range(m):
        for j in range(n):
            neuron_locs.append(np.array([i, j]))

    # iterate through each unit and find its neighbours on the map
    for j in range(m):
        for i in range(n):
            cneighbor_idxs = list()
            
            # Save the neighbours for a unit with location i, j
            if (i > 0):         
                cneighbor_idxs.append(j * n + i - 1)
            if (i < n - 1):
                cneighbor_idxs.append(j * n + i + 1)
            if (j > 0):
                cneighbor_idxs.append(j * n + i - n)
            if (j < m - 1):
                cneighbor_idxs.append(j * n + i + n)

            # Get the weights of the neighbouring units
            cneighbor_weights = weights[cneighbor_idxs]

            # Get the average distance between unit i, j and all of its neighbors
            # Expand dims to broadcast to each of the neighbors
            umatrix[j * n + i] = distance_matrix(np.expand_dims(weights[j * n + i], 0), cneighbor_weights).mean()

    bmu_indices = som.bmu_indices(tf.constant(input_data, dtype=tf.float32))

    return umatrix, bmu_indices


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    graph = tf.Graph()
    with graph.as_default():
        # Make sure you allow_soft_placement, some ops have to be put on the CPU (e.g. summary operations)
        session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False))

        num_inputs = 1024
        dims = 10
        clusters = 3
        # Makes toy clusters with pretty clear separation, see the sklearn site for more info
        blob_data = make_blobs(num_inputs, dims, clusters)
        # Scale the blob data for easier training. Also index 0 because the output is a (data, label) tuple.
        scaler = StandardScaler()
        input_data = scaler.fit_transform(blob_data[0])
        batch_size = 128

        # Build the TensorFlow dataset pipeline per the standard tutorial.
        dataset = tf.data.Dataset.from_tensor_slices(input_data.astype(np.float32))
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        next_element = iterator.get_next()

        # This is more neurons than you need but it makes the visualization look nicer
        m = 20
        n = 20

        # Build the SOM object and place all of its ops on the graph
        som = SelfOrganizingMap(m=m, n=n, dim=dims, max_epochs=20, gpus=1, session=session, graph=graph,
                                input_tensor=next_element, batch_size=batch_size, initial_learning_rate=0.1)
        # Initialize PCA weights
        som.pca_weights_init(input_data)
        init_op = tf.compat.v1.global_variables_initializer()
        session.run([init_op])

        # Note that I don't pass a SummaryWriter because I don't really want to record summaries in this script
        # If you want Tensorboard support just make a new SummaryWriter and pass it to this method
        som.train(num_inputs=num_inputs)

        weights = som.output_weights
        
        umatrix, bmu_loc = get_umatrix_optimized(input_data,weights, m, n)
        fig = plt.figure()
        plt.imshow(umatrix.reshape((m, n)), origin='lower')
        plt.show(block=True)
