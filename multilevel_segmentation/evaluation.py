"""
Evaluation Module

This module contains functions for evaluating segmentation results.
"""

import numpy as np
from typing import Dict, Optional, Union, List

from .utils import calculate_mse, calculate_psnr, jaccard_index


def evaluate_segmentation(
    ground_truth: np.ndarray,
    segmentation: np.ndarray,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate segmentation results against ground truth.
    
    Parameters
    ----------
    ground_truth : np.ndarray
        Ground truth segmentation.
    segmentation : np.ndarray
        Segmentation result to evaluate.
    metrics : List[str], optional
        List of metrics to compute. If None, computes all available metrics.
        
    Returns
    -------
    Dict[str, float]
        Dictionary of computed metrics.
    """
    if metrics is None:
        metrics = ['mse', 'psnr', 'jaccard', 'dice']
    
    results = {}
    
    # Convert images to binary if needed
    if ground_truth.dtype != bool:
        gt_binary = ground_truth > 0.5
    else:
        gt_binary = ground_truth
        
    if segmentation.dtype != bool:
        seg_binary = segmentation > 0.5
    else:
        seg_binary = segmentation
    
    # Compute requested metrics
    if 'mse' in metrics:
        results['mse'] = calculate_mse(ground_truth, segmentation)
        
    if 'psnr' in metrics:
        results['psnr'] = calculate_psnr(ground_truth, segmentation)
        
    if 'jaccard' in metrics:
        results['jaccard'] = jaccard_index(gt_binary, seg_binary)
        
    if 'dice' in metrics:
        intersection = np.logical_and(gt_binary, seg_binary).sum()
        dice = 2 * intersection / (gt_binary.sum() + seg_binary.sum())
        results['dice'] = dice if not np.isnan(dice) else 0.0
    
    if 'accuracy' in metrics:
        correct = np.sum(gt_binary == seg_binary)
        total = gt_binary.size
        results['accuracy'] = correct / total
        
    if 'precision' in metrics:
        true_positive = np.logical_and(gt_binary, seg_binary).sum()
        predicted_positive = seg_binary.sum()
        results['precision'] = true_positive / predicted_positive if predicted_positive > 0 else 0.0
        
    if 'recall' in metrics:
        true_positive = np.logical_and(gt_binary, seg_binary).sum()
        actual_positive = gt_binary.sum()
        results['recall'] = true_positive / actual_positive if actual_positive > 0 else 0.0
    
    return results


def compare_methods(
    ground_truth: np.ndarray,
    method_results: Dict[str, np.ndarray],
    metrics: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare different segmentation methods.
    
    Parameters
    ----------
    ground_truth : np.ndarray
        Ground truth segmentation.
    method_results : Dict[str, np.ndarray]
        Dictionary of segmentation results for different methods.
    metrics : List[str], optional
        List of metrics to compute. If None, computes all available metrics.
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary of evaluation metrics for each method.
    """
    results = {}
    
    for method_name, segmentation in method_results.items():
        results[method_name] = evaluate_segmentation(ground_truth, segmentation, metrics)
    
    return results


def find_optimal_parameters(
    image: np.ndarray,
    ground_truth: np.ndarray,
    model_class,
    parameter_grid: Dict[str, List],
    metric: str = 'jaccard'
) -> Dict:
    """
    Find optimal parameters for a segmentation model.
    
    Parameters
    ----------
    image : np.ndarray
        Input image to segment.
    ground_truth : np.ndarray
        Ground truth segmentation.
    model_class : class
        The segmentation model class to use.
    parameter_grid : Dict[str, List]
        Dictionary of parameters to search over.
    metric : str, optional
        Metric to optimize. Default is 'jaccard'.
        
    Returns
    -------
    Dict
        Dictionary containing optimal parameters and the best score.
    """
    best_score = -float('inf') if metric != 'mse' else float('inf')
    best_params = None
    
    # Generate all parameter combinations
    param_combinations = [{}]
    for param, values in parameter_grid.items():
        new_combinations = []
        for combo in param_combinations:
            for value in values:
                new_combo = combo.copy()
                new_combo[param] = value
                new_combinations.append(new_combo)
        param_combinations = new_combinations
    
    for params in param_combinations:
        # Create and train model with current parameters
        model = model_class(**params)
        result = model.segment(image)
        
        # Evaluate model
        metrics = evaluate_segmentation(ground_truth, result['segmented_image'], [metric])
        current_score = metrics[metric]
        
        # Check if better than current best
        if (metric == 'mse' and current_score < best_score) or \
           (metric != 'mse' and current_score > best_score):
            best_score = current_score
            best_params = params
    
    return {
        'params': best_params,
        'score': best_score
    }