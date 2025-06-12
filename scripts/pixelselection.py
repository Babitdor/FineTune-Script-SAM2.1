"""
Methods to select pixel coordinates from a segmentation mask
for a specific class with a selection algorithm like: fibonacci, laplace or gaussian.

How to use:
    # Select 10 pixels from class 1 using each method.
    coords_fib = select_pixels(seg_mask, class_id=1, num_points=10, selection_method='fibonacci')
    coords_laplace = select_pixels(seg_mask, class_id=1, num_points=10, selection_method='laplace')
    coords_gaussian = select_pixels(seg_mask, class_id=1, num_points=10, selection_method='gaussian')
    coords_shuffle = select_pixels(seg_mask, class_id=1, num_points=10, selection_method='shuffle')
    
    print("Fibonacci selection:\n", coords_fib)
    print("Laplace selection:\n", coords_laplace)
    print("Gaussian selection:\n", coords_gaussian)

"""

import numpy as np
from scipy.ndimage import center_of_mass, distance_transform_edt, label
from sklearn.cluster import KMeans


def select_pixels(segmentation_mask, class_id, num_points, selection_method='laplace'):
    """
    Select num_points pixel coordinates from segmentation_mask that belong to class_id,
    using one of three selection methods: 'fibonacci', 'laplace' or 'gaussian'.
    
    Parameters:
        segmentation_mask (np.ndarray): A 2D array (e.g. height x width) with class IDs.
        class_id (int): The class ID to select pixels from.
        num_points (int): The number of pixel coordinates to return.
        selection_method (str): The selection method to use ('fibonacci', 'centroid', 'fps', 'laplace', 'shuffle' or 'gaussian').
    
    Returns:
        np.ndarray: Array of shape (num_points, 2), where each row is an [x, y] coordinate.
    """
    # Find pixel locations (row, col) where the mask equals class_id.
    # Note: np.where returns (rows, cols), which correspond to (y, x)
    ys, xs = np.where(segmentation_mask == class_id)
    
    if len(xs) < num_points:
        raise ValueError(f"Not enough pixels for class {class_id}: found {len(xs)} but need {num_points}.")
    
    # Create an array of candidate coordinates in (x, y) order.
    candidates = np.column_stack((xs, ys))
    
    # For consistency, sort candidates (first by y then by x)
    # np.lexsort uses the last key first, so we sort with keys (x, y)
    candidates = candidates[np.lexsort((candidates[:, 0], candidates[:, 1]))]
    candidate_count = candidates.shape[0]
    
    # Choose the selection method.
    if selection_method.lower() == 'fibonacci':
        selected_idx = fibonacci_selection(candidate_count, num_points)
    elif selection_method.lower() == 'laplace':
        selected_idx = laplace_selection(candidate_count, num_points)
    elif selection_method.lower() == 'gaussian':
        selected_idx = gaussian_selection(candidate_count, num_points)
    elif selection_method.lower() == 'shuffle':
        return shuffle_selection(segmentation_mask, class_id, num_points)
    elif selection_method.lower() == 'centroid':
        return select_centroid(segmentation_mask, class_id, num_points)
    elif selection_method.lower() == 'kmeans':
        return select_kmeans(segmentation_mask, class_id, num_points)
    elif selection_method.lower() == 'fps':
        return select_instance_fps(segmentation_mask, class_id, num_points)
    else:
        raise ValueError("Unknown selection method. Choose 'fibonacci', 'laplace', 'shuffle', or 'gaussian'.")
    
    # Return the selected (x, y) coordinates.
    return candidates[selected_idx]


def fibonacci_selection(candidate_count, num_points):
    """
    Select indices from 0 to candidate_count-1 based on the Fibonacci sequence.
    We generate Fibonacci numbers and take them modulo candidate_count until we have
    num_points unique indices.
    """
    fib_indices = []
    a, b = 0, 1
    while len(fib_indices) < num_points:
        mod_index = a % candidate_count
        if mod_index not in fib_indices:
            fib_indices.append(mod_index)
        a, b = b, a + b
    return np.array(fib_indices)


def laplace_selection(candidate_count, num_points):
    """
    Select indices using a Laplace (double exponential) distribution.
    We weight indices with a Laplace PDF centered at candidate_count/2.
    """
    indices = np.arange(candidate_count)
    mu = candidate_count / 2.0
    scale = candidate_count / 6.0  # Adjust scale as needed.
    # Compute the Laplace probability density for each index.
    pdf = np.exp(-np.abs(indices - mu) / scale) / (2 * scale)
    pdf /= pdf.sum()  # Normalize to sum to 1.
    # Sample without replacement using the computed weights.
    selected = np.random.choice(indices, size=num_points, replace=False, p=pdf)
    return selected


def gaussian_selection(candidate_count, num_points):
    """
    Select indices using a Gaussian (normal) distribution.
    We weight indices with a Gaussian PDF centered at candidate_count/2.
    """
    indices = np.arange(candidate_count)
    mu = candidate_count / 2.0
    sigma = candidate_count / 6.0  # Adjust sigma as needed.
    pdf = np.exp(-0.5 * ((indices - mu) / sigma) ** 2)
    pdf /= pdf.sum()  # Normalize to sum to 1.
    selected = np.random.choice(indices, size=num_points, replace=False, p=pdf)
    return selected


def shuffle_selection(mask: np.ndarray, class_id: int, num_pixels: int) -> np.ndarray:
    """
    Selects pixel coordinates from a class mask area defined by class_id.
    Ensures selected pixels are well distributed based on a computed distance.

    :param mask: np.ndarray - The class mask (H, W) where each pixel has a class ID.
    :param class_id: int - The class ID for which pixel coordinates should be extracted.
    :param num_pixels: int - Number of pixels to select.
    :return: np.ndarray - Selected pixel coordinates in absolute format [[x1, y1], [x2, y2], ...]
    """
    # Get all coordinates where mask == class_id
    coords = np.argwhere(mask == class_id)
    
    # If there are no pixels of the given class_id, return an empty array
    if len(coords) == 0:
        return np.array([])

    np.random.shuffle(coords)  # This shuffles the array in place

    # Compute selection distance
    total_pixels = len(coords)
    if total_pixels <= num_pixels:
        # If not enough pixels, return all available pixels
        selected_coords = coords
    else:
        distance = max(1, total_pixels // num_pixels)  # Ensure distance is at least 1
        selected_coords = coords[::distance][:num_pixels]  # Select pixels with step size
    
    # Convert (row, col) to (x, y) format
    selected_coords = np.array([[y, x] for x, y in selected_coords])
    
    return selected_coords


def select_centroid(segmentation_mask, class_id, num_points):
    """
    Selects positive points for a given class in a segmentation mask.
    
    Parameters:
        segmentation_mask (np.ndarray): A 2D array where each pixel's value represents a class label.
        class_id (int or float): The class id for which positive points are to be selected.
        num_points (int): The total number of points to return.
    
    Returns:
        np.ndarray: An array of shape (num_points, 2) where each row is [x, y] coordinates.
                  x corresponds to the column and y to the row in the mask.
    """
    # Create a binary mask for the specified class.
    binary_mask = (segmentation_mask == class_id)
    
    # If no pixels match the class, return an empty array.
    if np.sum(binary_mask) == 0:
        return np.empty((0, 2))
    
    # Compute connected components to separate individual instances.
    labeled_mask, num_features = label(binary_mask)
    
    centroids = []
    # Compute the centroid for each connected component (instance).
    for instance_id in range(1, num_features + 1):
        com = center_of_mass(binary_mask, labeled_mask, instance_id)  # (row, col) as float
        # Convert to (x, y) where x = col, y = row.
        centroids.append([com[1], com[0]])
    centroids = np.array(centroids)
    
    # If we have equal or more centroids than requested, sample among them.
    if len(centroids) >= num_points:
        # Randomly choose 'num_points' centroids and return.
        indices = np.random.choice(len(centroids), num_points, replace=False)
        return centroids[indices]
    
    # Otherwise, we need to add extra points.
    additional_needed = num_points - len(centroids)
    # Find all (row, col) coordinates that belong to the class.
    all_coords = np.argwhere(binary_mask)  # (row, col)
    
    # For proper comparison, convert centroids to integer coordinates (rounding)
    # and change order to (row, col) for the comparison.
    centroids_rc = np.round(centroids[:, ::-1]).astype(int)
    centroid_set = {tuple(coord) for coord in centroids_rc}
    
    # Filter all_coords, removing coordinates already in centroids.
    extra_coords = np.array([tuple(coord) for coord in all_coords if tuple(coord) not in centroid_set])
    
    # If there are fewer extra coordinates than needed, use them all.
    if len(extra_coords) < additional_needed:
        additional_points = extra_coords
    else:
        indices_extra = np.random.choice(len(extra_coords), additional_needed, replace=False)
        additional_points = extra_coords[indices_extra]
    
    # Convert additional points from (row, col) to (x, y) i.e. (col, row).
    additional_points_xy = np.column_stack([additional_points[:, 1], additional_points[:, 0]])
    
    # Combine the centroids and the additional points.
    combined_points = np.vstack([centroids, additional_points_xy])
    
    # If combined_points accidentally exceeds num_points, sample to select exactly num_points.
    if combined_points.shape[0] > num_points:
        indices_comb = np.random.choice(combined_points.shape[0], num_points, replace=False)
        combined_points = combined_points[indices_comb]
    
    return combined_points


def select_kmeans(segmentation_mask, class_id, num_points, random_state=0):
    """
    Selects well‑distributed positive points for a given class by clustering the mask pixels.
    
    Parameters:
        segmentation_mask (np.ndarray): 2D array where each pixel’s value is a class label.
        class_id (int): the class for which to sample points.
        num_points (int): number of points to return.
        random_state (int): seed for KMeans reproducibility.
    
    Returns:
        np.ndarray of shape (M, 2), with M = min(num_points, #pixels),
        each row = [x, y]  (x=col, y=row).
    """    
    # 1) Extract pixel coordinates for this class
    mask = (segmentation_mask == class_id)
    coords = np.argwhere(mask)  # shape (P, 2): [row, col]
    if coords.size == 0:
        return np.empty((0, 2), dtype=int)

    # 2) If fewer pixels than num_points, just shuffle & return them
    P = coords.shape[0]
    if P <= num_points:
        rng = np.random.RandomState(random_state)
        sel = rng.permutation(P)[:num_points]
        chosen = coords[sel]
    else:
        # 3) K‑means clustering on the pixel coords
        kmeans = KMeans(n_clusters=num_points, random_state=random_state)
        kmeans.fit(coords)
        centers = kmeans.cluster_centers_  # floats (row, col)

        # 4) Snap each center to nearest actual mask pixel
        chosen = []
        for cy, cx in centers:
            d2 = (coords[:, 0] - cy)**2 + (coords[:, 1] - cx)**2
            idx = np.argmin(d2)
            chosen.append(coords[idx])
        chosen = np.array(chosen, dtype=int)

    # 5) Convert to [x, y] (col, row), then ensure contiguous
    pts = chosen[:, ::-1]           # this is a view with reversed stride
    return np.ascontiguousarray(pts)


def _farthest_point_sampling(coords, k, rng):
    """
    classic FPS / Poisson-disk on integer coordinates (row, col)
    returns indices into *coords*
    """
    N = coords.shape[0]
    if k >= N:
        return np.arange(N, dtype=int)

    sel = [rng.integers(N)]           # random seed to avoid bias
    d2 = np.full(N, np.inf)

    for _ in range(1, k):
        diff = coords - coords[sel[-1]]
        d2 = np.minimum(d2, np.sum(diff**2, axis=1))
        sel.append(int(np.argmax(d2)))

    return np.asarray(sel, dtype=int)


def select_instance_fps(seg_mask,
                        class_id: int,
                        num_points: int,
                        *,
                        area_power: float = 1.0,
                        min_area: int = 20,
                        rng: np.random.Generator | None = None
                        ) -> np.ndarray:
    """
    Instance-aware farthest-point sampler with speckle removal.

    Parameters
    ----------
    seg_mask   : (H, W) np.uint8   semantic mask
    class_id   : int               class we are sampling
    num_points : int               exact number of positive points wanted
    area_power : float             how much to favour big objects
    min_area   : int               discard components smaller than this
    rng        : np.random.Generator | None

    Returns
    -------
    pts : (num_points, 2) int32    [[x, y], …]   (x = col, y = row)
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1) binary mask for the class
    binary = seg_mask == class_id
    if not binary.any():
        # no pixels of this class → return an empty array
        return np.empty((0, 2), dtype=np.int32)

    # 2) connected components
    inst_map, n_inst = label(binary)

    # 3) collect instances and filter speckles
    coords_by_inst, areas = [], []
    for idx in range(1, n_inst + 1):
        coords = np.argwhere(inst_map == idx)      # (rows, cols)
        a = coords.shape[0]
        if a >= min_area:
            coords_by_inst.append(coords)
            areas.append(a)

    if not coords_by_inst:                         # everything was a speckle
        return np.empty((0, 2), dtype=np.int32)

    areas = np.asarray(areas, float)
    inst_count = len(coords_by_inst)

    # 4) select at most num_points largest instances
    order = np.argsort(-areas)                     # descending by area
    keep = order[:min(inst_count, num_points)]
    coords_by_inst = [coords_by_inst[i] for i in keep]
    areas = areas[keep]
    M = len(coords_by_inst)                        # 1 ≤ M ≤ num_points

    # 5) allocate points: at least 1 each, remainder ∝ area^area_power
    alloc = np.ones(M, dtype=int)
    left = num_points - M
    if left > 0:
        w = areas**area_power
        w /= w.sum()
        alloc += rng.multinomial(left, w)

    # 6) sample inside every kept instance
    all_pts = []
    for coords_rc, k in zip(coords_by_inst, alloc):
        if k == 1 or coords_rc.shape[0] <= 30:     # tiny blob → centroid
            cy, cx = coords_rc.mean(axis=0)
            chosen = np.array([[int(cx + 0.5), int(cy + 0.5)]], dtype=int)

        else:
            # distance-transform seed (pixel farthest from boundary)
            mask_inst = np.zeros_like(binary, bool)
            mask_inst[tuple(coords_rc.T)] = True
            seed_rc = np.unravel_index(np.argmax(distance_transform_edt(mask_inst)),
                                       mask_inst.shape)
            seed_idx = np.flatnonzero((coords_rc == seed_rc).all(1))[0]

            idxs = _farthest_point_sampling(coords_rc, k, rng)
            if seed_idx not in idxs:               # guarantee central seed
                idxs[0] = seed_idx
            chosen = coords_rc[idxs][:, ::-1]      # to (x, y)

        all_pts.append(chosen)

    pts = np.vstack(all_pts)

    # numerical safety (should already be exact)
    if pts.shape[0] > num_points:
        pts = pts[rng.choice(pts.shape[0], num_points, replace=False)]

    return pts.astype(np.int32, copy=False)
