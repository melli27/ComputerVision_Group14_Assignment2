# TUWIEN - CV: Task3 - Scene recognition using Bag of Visual Words
# *********+++++++++*******++++ GROUP NO. 14
from typing import List
import sklearn
import sklearn.metrics.pairwise as sklearn_pairwise
import cv2
import numpy as np
import random
import time


def extract_dsift(images: List[np.ndarray], stepsize: int, num_samples: int = None) -> List[np.ndarray]:
    """
    Extracts dense feature points on a regular grid with 'stepsize' and optionally returns
    'num_samples' random samples per image. If 'num_samples' is not provided, it takes all
    features extracted with the given 'stepsize'. SIFT.compute has the argument "keypoints",
    which should be set to a list of keypoints for each square.
    
    Args:
    - images (List[np.ndarray]): List of images to extract dense SIFT features [num_of_images x n x m] - float
    - stepsize (int): Grid spacing, step size in x and y direction.
    - num_samples (int, optional): Random number of samples per image.

    Returns:
    - List[np.ndarray]: SIFT descriptors for each image [number_of_images x num_samples x 128] - float
    """
    tic = time.perf_counter()

    # student_code start

    # Create SIFT
    sift = cv2.SIFT_create()
    all_descriptors = []

    for img in images: 
        # Open CV needs uint8 (0-255):
        img = (img * 255).astype(np.uint8)
        h, w = img.shape[:2]
        
        # Set of keypoints for each square 
        keypoints = []

        # Create Grid
        for y in range(stepsize, h, stepsize):
            for x in range(stepsize, w, stepsize):
                keypoints.append(cv2.KeyPoint(float(x), float(y), float(stepsize)))

        if num_samples != None:
            if len(keypoints) > num_samples:
                keypoints = random.sample(keypoints, num_samples)

        
        _, descriptors = sift.compute(img, keypoints)

        all_descriptors.append(descriptors)

    # student_code end

    toc = time.perf_counter()
    print("DSIFT Extraction:", toc - tic, " seconds")

    # all_descriptors : list sift descriptors per image [number_of_images x num_samples x 128] - float
    return all_descriptors


def count_visual_words(dense_feat: List[np.ndarray], centroids: List[np.ndarray]) -> List[np.ndarray]:
    """
    For classification, generates a histogram of word occurrences per image.
    Utilizes sklearn_pairwise.pairwise_distances(..) to assign the descriptors per image
    to the nearest centroids and counts the occurrences of each centroid. The histogram
    should be as long as the vocabulary size (number of centroids).

    Args:
    - dense_feat (List[np.ndarray]): List of SIFT descriptors per image [number_of_images x num_samples x 128] - float
    - centroids (List[np.ndarray]): Centroids of clusters [vocabulary_size x 128]

    Returns:
    - List[np.ndarray]: List of histograms per image [number_of_images x vocabulary_size]
    """
    tic = time.perf_counter()

    # student_code start
    histograms = []

    # Per image
    for descriptors in dense_feat:
        # Compute euclidean disntance between local descriptor in the image and all cluster centroids (visual words) in the vocabularly
        distances = sklearn_pairwise.pairwise_distances(descriptors, centroids)
        # Assign each descriptor to its nearest visual word/centroid
        nearest_centroid_indices = np.argmin(distances, axis=1)
        # Create histogram to count how often each visual word appears in the img
        # Bins are set to given range to ensure every word has a slot
        hist, _ = np.histogram(nearest_centroid_indices, bins=range(len(centroids) + 1))
        
        histograms.append(hist)
     

    # student_code end

    toc = time.perf_counter()
    print("Counting visual words:", toc - tic, " seconds")

    # histograms : list of histograms per image [number_of_images x vocabulary_size]
    return histograms


def calculate_vlad_descriptors(dense_feat: List[np.ndarray], centroids: List[np.ndarray]) -> List[np.ndarray]:
    """
    For classification, generate a histogram of word occurence per image
     Use sklearn_pairwise.pairwise_distances(..) to assign the descriptors per image
     to the nearest centroids and calculate for each word the residual to the nearest centroid
     The final feature vector should be as long as the vocabulary size (number of centroids) x feature dimension
     L2-normalize the final descriptors via sklearn.preprocessing.normalize.
     
    Args:
    - dense_feat : list sift descriptors per image [number_of_images x num_samples x 128] - float
    - centroids : centroids of clusters [vocabulary_size x 128]

    Returns:
    - List[np.ndarray]: List of histograms per image [number_of_images x (vocabulary_size x feature dimension)]
    """
    tic = time.perf_counter()

    
    # student_code start
    image_descriptors = []
    for descriptors in dense_feat:
        # Find nearest centroid for each descriptor
        distances = sklearn_pairwise.pairwise_distances(descriptors, centroids)
        nearest_centroid_indices = np.argmin(distances, axis=1)
        # Determine residual = x - c, which is the differencebetween feature point and cluster center
        # Represents the vector pointing from the centroid to actual descriptor whtich same shape as descriptor
        residuals = descriptors - centroids[nearest_centroid_indices]
        
        # Aggregation: sum residuals for each visual word/cluster
        # Mathematical formulation for a visual word is: 
        # vlad = sum{i=1}{N} alpha_k(x_i) * (x_i - c_k) with x_i is i-th local descr. and c_k k-th cluster centroid
        # alpha acts as a Schalter: If the current feature x_i part of cluster c_k, then is alpha_k = 1. 
        # Otherwise, alpha_k = 0 and the feature will be ignored.

        # Create empty vlad matrix
        vlad = np.zeros(centroids.shape)
        # Use add.at to perform unbuffered in place addition
        # For each index in nearest_centroid_indices it adds corresponding residual to the specific row in vlad matrix
        np.add.at(vlad, nearest_centroid_indices, residuals)

        # Concatenate 
        # convert KxD matrix into a single 1D vector of size K*D
        vlad = vlad.flatten().reshape(1, -1)
        # Final global descriptor is l2-normalized
        # Descriptor is invariant to the number of descriptors in the img
        V = sklearn.preprocessing.normalize(vlad, norm="l2")
        image_descriptors.append(V.flatten())
     
    # student_code end
    

    toc = time.perf_counter()
    print("Counting visual words:", toc - tic, " seconds")

    # histograms : list of histograms per image [number_of_images x vocabulary_size]
    return image_descriptors