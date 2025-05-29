"""
Comparative Analysis

This example performs a comparative analysis between different segmentation approaches
and visualizes the results along with quantitative metrics.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

from multilevel_segmentation import SegmentationModel
from multilevel_segmentation.combined_method import CombinedMethod
from multilevel_segmentation.evaluation import compare_methods
from multilevel_segmentation.visualization import compare_methods as visualize_comparison

# Create a synthetic image with ground truth
def create_synthetic_image(size=300, noise_level=0.1):
    # Create ground truth
    ground_truth = np.zeros((size, size), dtype=np.uint8)
    
    # Circle in the center
    for i in range(size):
        for j in range(size):
            if (i - size//2)**2 + (j - size//2)**2 < (size//4)**2:
                ground_truth[i, j] = 1
    
    # Rectangle at the top-left
    ground_truth[size//6:2*size//6, size//6:5*size//6] = 2
    
    # Create image from ground truth with noise
    img = np.zeros((size, size), dtype=np.float32)
    img[ground_truth == 1] = 0.7
    img[ground_truth == 2] = 0.4
    
    # Add noise
    np.random.seed(42)
    img += noise_level * np.random.randn(size, size)
    img = np.clip(img, 0, 1)
    
    return img, ground_truth

# Create synthetic images with different noise levels
print("Creating synthetic images with different noise levels...")
noise_levels = [0.05, 0.1, 0.2]
images = []
ground_truths = []

for noise in noise_levels:
    img, gt = create_synthetic_image(noise_level=noise)
    images.append(img)
    ground_truths.append(gt)

# Define segmentation methods to compare
methods = {
    'K-means': SegmentationModel(method='clustering', clustering_algorithm='kmeans', clusters=3),
    'Fuzzy C-means': SegmentationModel(method='clustering', clustering_algorithm='fuzzy_cmeans', clusters=3),
    'Two-Step': SegmentationModel(method='two_step', clusters=3),
    'Combined (Default)': SegmentationModel(method='combined', clusters=3, iterations=5),
    'Combined (Custom)': CombinedMethod(n_clusters=3, n_iterations=5, feedback_strength=0.7, 
                                        morphological_sequence=['closing', 'opening'])
}

# Perform comparison for each noise level
print("Performing segmentation and evaluation...")
all_metrics = []
execution_times = {}

for i, (img, gt) in enumerate(zip(images, ground_truths)):
    print(f"\nProcessing image with noise level {noise_levels[i]}")
    
    # Apply each method and measure execution time
    method_results = {}
    
    for name, method in methods.items():
        start_time = time.time()
        result = method.segment(img)
        end_time = time.time()
        
        method_results[name] = result['segmented_image']
        
        if name not in execution_times:
            execution_times[name] = []
        
        execution_times[name].append(end_time - start_time)
        print(f"  {name} - Execution time: {end_time - start_time:.3f} seconds")
    
    # Compare segmentation results
    metrics = compare_methods(gt, method_results, metrics=['jaccard', 'dice', 'accuracy'])
    all_metrics.append(metrics)
    
    # Visualize results
    plt.figure(figsize=(20, 10))
    
    # Original and ground truth
    plt.subplot(2, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Original (Noise: {noise_levels[i]})")
    plt.axis('off')
    
    plt.subplot(2, 4, 5)
    plt.imshow(gt, cmap='viridis')
    plt.title("Ground Truth")
    plt.axis('off')
    
    # Results for each method
    for j, (name, result) in enumerate(method_results.items(), 1):
        if j + 1 <= 4:
            plt.subplot(2, 4, j + 1)
        else:
            plt.subplot(2, 4, j + 3)
            
        plt.imshow(result, cmap='viridis')
        
        # Add metrics to title
        metrics_str = ""
        for metric, value in all_metrics[i][name].items():
            metrics_str += f"{metric}: {value:.3f}, "
        metrics_str = metrics_str[:-2]  # Remove trailing comma and space
        
        plt.title(f"{name}\n{metrics_str}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"comparison_noise_{noise_levels[i]}.png")
    plt.close()

# Plot performance metrics across noise levels
metrics_to_plot = ['jaccard', 'dice', 'accuracy']
for metric in metrics_to_plot:
    plt.figure(figsize=(10, 6))
    
    for method_name in methods.keys():
        metric_values = [metrics[method_name][metric] for metrics in all_metrics]
        plt.plot(noise_levels, metric_values, marker='o', label=method_name)
    
    plt.xlabel('Noise Level')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} vs Noise Level')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"metric_{metric}_vs_noise.png")
    plt.close()

# Plot execution times
plt.figure(figsize=(10, 6))
for method_name, times in execution_times.items():
    plt.plot(noise_levels, times, marker='o', label=method_name)

plt.xlabel('Noise Level')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs Noise Level')
plt.legend()
plt.grid(True)
plt.savefig("execution_time_vs_noise.png")
plt.close()

print("\nComparative analysis completed! Results saved as PNG files.")