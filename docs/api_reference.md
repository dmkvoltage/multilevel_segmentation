# API Reference

## Core Classes

### `SegmentationModel`

The main class for performing image segmentation using various methods.

```python
from multilevel_segmentation import SegmentationModel

# Create a model with default parameters
model = SegmentationModel()

# Create a model with custom parameters
model = SegmentationModel(
    method='combined',
    clusters=3,
    iterations=5,
    clustering_algorithm='kmeans',
    morphological_operations=['opening', 'closing']
)

# Segment an image
result = model.segment(image)
```

#### Parameters

- `method` (str): The segmentation method to use. Options are 'clustering', 'two_step', or 'combined'. Default is 'combined'.
- `clusters` (int): The number of clusters to use. Default is 3.
- `iterations` (int): The number of iterations for the combined approach. Default is 5.
- `clustering_algorithm` (str): The clustering algorithm to use. Options are 'kmeans' or 'fuzzy_cmeans'. Default is 'kmeans'.
- `morphological_operations` (list): List of morphological operations to apply in sequence. Default is ['opening', 'closing'].

#### Methods

- `segment(image: np.ndarray) -> Dict[str, np.ndarray]`: Segment the input image using the specified method.

#### Returns

The `segment` method returns a dictionary with the following keys:

- `segmented_image`: The segmented image.
- `labels`: The cluster labels for each pixel.
- `centers`: The cluster centers.
- `intermediate_results`: A list of intermediate results (for 'combined' method only).

### `CombinedMethod`

A class that implements our novel iterative approach combining clustering and morphological operations.

```python
from multilevel_segmentation.combined_method import CombinedMethod

# Create a model with default parameters
model = CombinedMethod()

# Create a model with custom parameters
model = CombinedMethod(
    n_clusters=3,
    n_iterations=5,
    clustering_algorithm='kmeans',
    morphological_sequence=['opening', 'closing'],
    feedback_strength=0.5
)

# Segment an image
result = model.segment(image)
```

#### Parameters

- `n_clusters` (int): Number of clusters. Default is 3.
- `n_iterations` (int): Number of iterations. Default is 5.
- `clustering_algorithm` (str): Clustering algorithm to use. Options are 'kmeans' or 'fuzzy_cmeans'. Default is 'kmeans'.
- `morphological_sequence` (list): Sequence of morphological operations to apply in each iteration. Default is ['opening', 'closing'].
- `feedback_strength` (float): Strength of the feedback between iterations (0 to 1). Default is 0.5.

#### Methods

- `segment(image: np.ndarray) -> Dict[str, np.ndarray]`: Segment the input image using the combined method.

## Clustering Functions

### `kmeans`

```python
from multilevel_segmentation.clustering import kmeans

labels, centers = kmeans(image, n_clusters=3, max_iter=100, tol=1e-4)
```

#### Parameters

- `image` (np.ndarray): The input image to segment (grayscale).
- `n_clusters` (int): The number of clusters.
- `max_iter` (int): Maximum number of iterations. Default is 100.
- `tol` (float): Tolerance for convergence. Default is 1e-4.

#### Returns

- `labels` (np.ndarray): Cluster labels for each pixel.
- `centers` (np.ndarray): Cluster centers.

### `fuzzy_cmeans`

```python
from multilevel_segmentation.clustering import fuzzy_cmeans

labels, centers = fuzzy_cmeans(image, n_clusters=3, m=2.0, max_iter=100, tol=1e-4)
```

#### Parameters

- `image` (np.ndarray): The input image to segment (grayscale).
- `n_clusters` (int): The number of clusters.
- `m` (float): Fuzziness parameter. Default is 2.0.
- `max_iter` (int): Maximum number of iterations. Default is 100.
- `tol` (float): Tolerance for convergence. Default is 1e-4.

#### Returns

- `labels` (np.ndarray): Cluster labels for each pixel.
- `centers` (np.ndarray): Cluster centers.

## Morphological Operations

### `erosion`

```python
from multilevel_segmentation.morphology import erosion

eroded_image = erosion(image, se=None, se_shape='square', se_size=3)
```

#### Parameters

- `image` (np.ndarray): The input image.
- `se` (np.ndarray, optional): The structural element. If None, it will be created using se_shape and se_size.
- `se_shape` (str, optional): Shape of the structural element if se is None. Default is 'square'.
- `se_size` (int, optional): Size of the structural element if se is None. Default is 3.

#### Returns

- `eroded_image` (np.ndarray): The eroded image.

### `dilation`

```python
from multilevel_segmentation.morphology import dilation

dilated_image = dilation(image, se=None, se_shape='square', se_size=3)
```

### `opening`

```python
from multilevel_segmentation.morphology import opening

opened_image = opening(image, se=None, se_shape='square', se_size=3)
```

### `closing`

```python
from multilevel_segmentation.morphology import closing

closed_image = closing(image, se=None, se_shape='square', se_size=3)
```

## Visualization Functions

### `visualize_results`

```python
from multilevel_segmentation.visualization import visualize_results

visualize_results(
    original_image,
    segmentation_result,
    save_path="result.png",
    show=True,
    figsize=(15, 10)
)
```

#### Parameters

- `original_image` (np.ndarray): The original input image.
- `segmentation_result` (Dict[str, np.ndarray]): Dictionary containing segmentation results.
- `save_path` (str, optional): Path to save the visualization. If None, it won't be saved.
- `show` (bool, optional): Whether to display the visualization. Default is True.
- `figsize` (Tuple[int, int], optional): Figure size. Default is (15, 10).

### `compare_methods`

```python
from multilevel_segmentation.visualization import compare_methods

compare_methods(
    original_image,
    results,
    metrics=None,
    save_path="comparison.png",
    show=True
)
```

#### Parameters

- `original_image` (np.ndarray): The original input image.
- `results` (Dict[str, Dict[str, np.ndarray]]): Dictionary of segmentation results for different methods.
- `metrics` (Dict[str, Dict[str, float]], optional): Dictionary of evaluation metrics for different methods.
- `save_path` (str, optional): Path to save the visualization. If None, it won't be saved.
- `show` (bool, optional): Whether to display the visualization. Default is True.

## Evaluation Functions

### `evaluate_segmentation`

```python
from multilevel_segmentation.evaluation import evaluate_segmentation

metrics = evaluate_segmentation(
    ground_truth,
    segmentation,
    metrics=['mse', 'psnr', 'jaccard', 'dice']
)
```

#### Parameters

- `ground_truth` (np.ndarray): Ground truth segmentation.
- `segmentation` (np.ndarray): Segmentation result to evaluate.
- `metrics` (List[str], optional): List of metrics to compute. If None, computes all available metrics.

#### Returns

- `metrics` (Dict[str, float]): Dictionary of computed metrics.

### `compare_methods`

```python
from multilevel_segmentation.evaluation import compare_methods

metrics = compare_methods(
    ground_truth,
    method_results,
    metrics=['jaccard', 'dice']
)
```

#### Parameters

- `ground_truth` (np.ndarray): Ground truth segmentation.
- `method_results` (Dict[str, np.ndarray]): Dictionary of segmentation results for different methods.
- `metrics` (List[str], optional): List of metrics to compute. If None, computes all available metrics.

#### Returns

- `metrics` (Dict[str, Dict[str, float]]): Dictionary of evaluation metrics for each method.