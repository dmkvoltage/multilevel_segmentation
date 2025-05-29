"""
Advanced Usage Example

This example demonstrates advanced usage of the multilevel_segmentation library,
including parameter tuning, evaluation, and visualization of intermediate results.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from multilevel_segmentation import SegmentationModel
from multilevel_segmentation.combined_method import CombinedMethod
from multilevel_segmentation.evaluation import evaluate_segmentation, compare_methods
from multilevel_segmentation.visualization import visualize_intermediate_results

# Load or create an image
try:
    # Replace with your own image path
    image_path = "sample_image.jpg"
    img = np.array(Image.open(image_path))
except FileNotFoundError:
    # Create a synthetic image with ground truth
    print("Sample image not found, creating synthetic image with ground truth...")
    size = 200
    
    # Create ground truth segmentation (3 regions)
    ground_truth = np.zeros((size, size), dtype=np.uint8)
    
    # Region 1: Circle in the center
    for i in range(size):
        for j in range(size):
            if (i - size//2)**2 + (j - size//2)**2 < (size//4)**2:
                ground_truth[i, j] = 1
    
    # Region 2: Rectangle at the top-left
    ground_truth[size//6:2*size//6, size//6:5*size//6] = 2
    
    # Create corresponding image with noise
    img = np.zeros((size, size), dtype=np.float32)
    img[ground_truth == 1] = 0.7
    img[ground_truth == 2] = 0.4
    
    # Add noise
    np.random.seed(42)
    img += 0.1 * np.random.randn(size, size)
    img = np.clip(img, 0, 1)

# Try different clustering approaches
print("Testing different clustering algorithms with 3 clusters...")
kmeans_model = SegmentationModel(method='clustering', clusters=3, clustering_algorithm='kmeans')
fcm_model = SegmentationModel(method='clustering', clusters=3, clustering_algorithm='fuzzy_cmeans')

kmeans_result = kmeans_model.segment(img)
fcm_result = fcm_model.segment(img)

# Try different numbers of clusters
print("Testing different numbers of clusters...")
results_by_clusters = {}
for n_clusters in [2, 3, 4, 5]:
    model = SegmentationModel(method='combined', clusters=n_clusters, iterations=3)
    results_by_clusters[f"{n_clusters} clusters"] = model.segment(img)

# Experiment with the combined method parameters
print("Testing different combined method parameters...")
# 1. Default parameters
default_combined = CombinedMethod()
default_result = default_combined.segment(img)

# 2. More iterations
more_iterations = CombinedMethod(n_iterations=10)
more_iter_result = more_iterations.segment(img)

# 3. Different morphological sequence
diff_morphology = CombinedMethod(morphological_sequence=['dilation', 'erosion'])
diff_morph_result = diff_morphology.segment(img)

# 4. Stronger feedback
stronger_feedback = CombinedMethod(feedback_strength=0.8)
stronger_fb_result = stronger_feedback.segment(img)

# Visualize intermediate results
print("Visualizing intermediate results from combined method...")
plt.figure(figsize=(15, 10))
visualize_intermediate_results(
    img, 
    default_result['intermediate_results'],
    save_path="intermediate_steps.png",
    show=False
)

# Compare methods
if 'ground_truth' in locals():
    print("Evaluating segmentation quality...")
    method_results = {
        'K-means': kmeans_result['segmented_image'],
        'Fuzzy C-means': fcm_result['segmented_image'],
        'Combined (Default)': default_result['segmented_image'],
        'Combined (More Iterations)': more_iter_result['segmented_image']
    }
    
    metrics = compare_methods(ground_truth, method_results, ['jaccard', 'dice', 'accuracy'])
    
    print("\nSegmentation Metrics:")
    for method, values in metrics.items():
        print(f"{method}:")
        for metric, value in values.items():
            print(f"  {metric}: {value:.4f}")

print("Advanced example completed! Results saved to intermediate_steps.png")