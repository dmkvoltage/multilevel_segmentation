"""
Clustering Algorithms

This module contains implementations of clustering algorithms used for image segmentation,
implemented from scratch without using scikit-learn or other ML libraries.
"""

import numpy as np
from typing import Tuple, Optional


def kmeans(
    image: np.ndarray, 
    n_clusters: int,
    max_iter: int = 100, 
    tol: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implement K-means clustering algorithm from scratch.
    
    Parameters
    ----------
    image : np.ndarray
        The input image to segment (grayscale).
    n_clusters : int
        The number of clusters.
    max_iter : int, optional
        Maximum number of iterations. Default is 100.
    tol : float, optional
        Tolerance for convergence. Default is 1e-4.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing cluster labels and cluster centers.
    """
    # Reshape the image to a 1D array of feature vectors
    height, width = image.shape
    X = image.reshape((-1, 1))
    
    # Randomly initialize cluster centers
    centers = np.random.uniform(np.min(X), np.max(X), size=(n_clusters, 1))
    
    # Initialize labels
    labels = np.zeros(X.shape[0], dtype=int)
    
    for iteration in range(max_iter):
        # Assign each pixel to the nearest cluster
        distances = np.zeros((X.shape[0], n_clusters))
        
        for i in range(n_clusters):
            # Calculate Euclidean distance to each cluster center
            distances[:, i] = np.linalg.norm(X - centers[i], axis=1)
        
        # Assign to the closest cluster
        new_labels = np.argmin(distances, axis=1)
        
        # Check for convergence
        if np.sum(new_labels != labels) / len(labels) < tol:
            break
            
        labels = new_labels
        
        # Update cluster centers
        old_centers = centers.copy()
        for i in range(n_clusters):
            if np.sum(labels == i) > 0:
                centers[i] = np.mean(X[labels == i], axis=0)
        
        # Check for convergence based on center movement
        center_shift = np.linalg.norm(centers - old_centers)
        if center_shift < tol:
            break
    
    # Reshape labels back to image dimensions
    return labels.reshape(height, width), centers


def fuzzy_cmeans(
    image: np.ndarray, 
    n_clusters: int, 
    m: float = 2.0,
    max_iter: int = 100, 
    tol: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implement Fuzzy C-means clustering algorithm from scratch.
    
    Parameters
    ----------
    image : np.ndarray
        The input image to segment (grayscale).
    n_clusters : int
        The number of clusters.
    m : float, optional
        Fuzziness parameter. Default is 2.0.
    max_iter : int, optional
        Maximum number of iterations. Default is 100.
    tol : float, optional
        Tolerance for convergence. Default is 1e-4.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing cluster labels and cluster centers.
    """
    # Reshape the image to a 1D array of feature vectors
    height, width = image.shape
    X = image.reshape((-1, 1))
    n_samples = X.shape[0]
    
    # Randomly initialize membership matrix
    U = np.random.rand(n_samples, n_clusters)
    # Normalize so that sum of memberships for each pixel equals 1
    U = U / np.sum(U, axis=1, keepdims=True)
    
    for iteration in range(max_iter):
        # Compute cluster centers
        um = U ** m
        centers = np.zeros((n_clusters, 1))
        
        for i in range(n_clusters):
            numerator = np.sum(um[:, i].reshape(-1, 1) * X, axis=0)
            denominator = np.sum(um[:, i])
            centers[i] = numerator / denominator
        
        # Update membership matrix
        old_U = U.copy()
        
        for i in range(n_samples):
            distances = np.linalg.norm(X[i] - centers, axis=1)
            
            # Handle the case of zero distance (pixel exactly at center)
            if np.any(distances == 0):
                U[i, :] = 0
                U[i, distances == 0] = 1
            else:
                # Calculate new memberships
                dist_factor = distances ** (2 / (1 - m))
                U[i, :] = 1 / (np.sum(dist_factor.reshape(-1, 1) / dist_factor, axis=0))
        
        # Check for convergence
        if np.linalg.norm(U - old_U) < tol:
            break
    
    # Determine crisp labels from fuzzy memberships
    labels = np.argmax(U, axis=1).reshape(height, width)
    
    return labels, centers