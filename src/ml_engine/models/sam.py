"""
Custom SAM predictor for multi-class sparse point inputs. 
"""
import logging
import numpy
import time
import torch

from segment_anything import SamPredictor
from numpy.typing import NDArray
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Optional
)


logger = logging.getLogger("SAM")


class SparsePointsSamPredictor:
    """
    A custom Segment Anything Model (SAM) for handling multi-class sparse point inputs.
    See:

    * [Segment Anything GitHub Repo](https://github.com/facebookresearch/segment-anything/tree/main)

    Parameters
    ----------
    predictor : SamPredictor
        An instance of the 
        [SAM predictor](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/predictor.py) 
        from the `segment_anything` library.
    
    device : str, optional
        The device to use for the predictor. It can be either "cpu" or "cuda".
        Default is "cpu".
    """
    _SUPPORTED_DEVICES = {"cpu", "cuda"}
    _IMAGE_FORMATS = {"RGB", "BGR"}

    def __init__(self, predictor: SamPredictor, device: str = "cpu"):
        # Validations
        # ---------------------------------------------------
        # Check the input processing device
        device = device.lower().strip()
        if device not in self._SUPPORTED_DEVICES:
            raise ValueError(f"Invalid device. Possible values: {self._SUPPORTED_DEVICES}. Got: '{device}'")
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available. Using CPU instead")
            device = "cpu"


        # Public attributes
        # ---------------------------------------------------
        self.predictor = predictor
        self.device = device

        self.processing_times: Dict[str, float] = {}        # General workflow processing times in seconds
        self.prediction_times: Dict[str, float] = {}        # Per-class prediction times in seconds
        self.class_to_points: Dict[str, NDArray] = {}       # Mapping from class names to (x, y) coordinates

        # SAM's mask threshold for foreground detection (foreground = mask > threshold). 
        # It is used to determine whether a pixel is considered foreground for a class 
        # based on the raw logits. The default value is 0.0 (SAM's default), which 
        # corresponds to a sigmoid probability of 0.5.
        self.mask_threshold = float(getattr(self.predictor.model, "mask_threshold", 0.0))

        # Mask logits predicted. Shape: (n_classes, H, W)
        # self.mask_logits: torch.Tensor = torch.empty(0, 0, 0, device=self.device, dtype=torch.float32)
        
        # Private attributes
        # ---------------------------------------------------
        self._class_names: List[str] = []       # Class names for the points
        # self._image: Optional[NDArray] = None   # Image to predict on
        
        # Current image dimensions (height, width) and status for the input image
        self._img_height: int = 0
        self._img_width: int = 0
        self._img_set: bool = False

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"predictor={self.predictor.__class__.__name__}, "
            f"device={self.device}"
            ")"
        )

    def __repr__(self):
        return str(self)


    # Private methods
    # --------------------------------------------------------------
    def _reset_state(self):
        """Reset the predictor state and clear all stored data."""
        self.processing_times.clear()
        self.prediction_times.clear()
        self.class_to_points.clear()
        
        self._class_names.clear()
        # self._image = None
        self._img_height = 0
        self._img_width = 0
        self._img_set = False
    
    def _preprocess(
            self,
            image: NDArray,
            input_points: NDArray,
            target_labels: NDArray,
            sorted_classes: List[str],
            image_format: str
    ):
        # Compute mapping from class names to coordinates
        # TODO: What happens if there are no input points for a class?
        for idx, class_name in enumerate(sorted_classes):
            # Check if there are input points for the class
            if idx not in target_labels:
                logger.warning(f"No input points for class '{class_name}' (id={idx})")
                continue

            class_coords = input_points[target_labels == idx]
            self.class_to_points[class_name] = class_coords
            logger.debug(f"Total points for class '{class_name}': {len(class_coords)}")
        logger.info(f"Total input points: {len(input_points)}")

        # Set SAM embedding
        t0 = time.perf_counter()
        try:
            logger.info("Calculating the image embeddings")
            self.predictor.set_image(image=image, image_format=image_format)
            self._img_height, self._img_width = image.shape[:2]
            self._img_set = True
            # self._image = image
        except Exception as e:
            if self.device == "cuda":
                torch.cuda.empty_cache()
            raise ValueError(f"Fail to calculate image embedding: {e}.") from e

        # Update the class names
        self._class_names = sorted_classes
        
        # Update the processing time for embeddings
        embeddings_time = time.perf_counter() - t0
        self.processing_times["embeddings"] = embeddings_time
        logger.info(f"Image embeddings calculated in {embeddings_time:.3f} seconds")

    def _get_class_points(self, class_name: str, include_background: bool) -> Tuple[NDArray, NDArray]:
        """
        Get the needed SAM's input points (coordinates and labels) for a specific class, 
        optionally including background points.

        Parameters
        ----------
        class_name : str
            The name of the class for which to get the input points. 
            Must be present in ``self._class_names`` attribute.
        
        include_background : bool
            Whether to include background points in the input points.
        
        Returns
        -------
        Tuple[NDArray, NDArray]
            A tuple containing two elements:
            
            1.  ``input_points``: 2D integer array of shape (N, 2) containing the ``(x, y)`` 
                coordinates of the input points.
            
            2.  ``input_labels``: 1D integer array of shape (N,) containing the corresponding 
                labels of the input points. A value of ``1`` indicates a positive point (foreground), 
                while a value of ``0`` indicates a negative point (background).
        """
        if class_name not in self.class_to_points:
            input_points = numpy.array([])
            input_labels = numpy.array([])

        else:
            # Get positive class data
            positive_points = self.class_to_points[class_name]

            # Create the input points and labels
            input_points_list: List[NDArray] = []
            input_labels_list: List[NDArray] = []
            input_points_list.append(positive_points)
            input_labels_list.append(numpy.ones(shape=len(positive_points), dtype=int))

            # Include negative points in the input points prompt
            if include_background:
                # Ordered list (not a set) so the sampling is reproducible
                negative_list = [
                    self.class_to_points[negative_class]
                    for negative_class in self._class_names
                    if (negative_class != class_name) and (negative_class in self.class_to_points)
                ]

                # Include negative points
                if len(negative_list) > 0:
                    negative_points = numpy.vstack(negative_list)
                    input_points_list.append(negative_points)
                    input_labels_list.append(numpy.zeros(shape=len(negative_points), dtype=int))

                else:
                    logger.warning(f"('{class_name}') No background points available")

            # Create the inputs as arrays
            input_points = numpy.vstack(input_points_list)
            input_labels = numpy.concatenate(input_labels_list)

        return input_points, input_labels


    # Public methods
    # --------------------------------------------------------------
    def set_input(
            self,
            image: NDArray,
            input_points: NDArray,
            target_labels: NDArray,
            sorted_classes: List[str],
            image_format: str = "RGB"
    ):
        """
        Set the input data for the predictor.

        Parameters
        ----------
        image : NDArray
            The image to predict on. It should be a 3D uint8 array with shape (height, width, channels).

        input_points : NDArray
            2D array of input points with shape (N, 2) where N is the number of points.
            Each point should be represented as (x, y) pixel coordinates in the image.

        target_labels : NDArray
            1D array of class labels corresponding to the input points. Each label should be an
            integer representing the class index in `sorted_class_names`.

        sorted_classes : List[str]
            List of class names sorted in the order of their indices. The length of this list should
            match the number of unique labels in `target_labels`.

        image_format : str, optional
            The color format of the image channels. It can be either "RGB" or "BGR".
            Default is "RGB".
        """
        self._reset_state()
        
        # Check the image format
        image_format = image_format.upper().strip()
        if image_format not in self._IMAGE_FORMATS:
            raise ValueError(f"Invalid image format. Possible values: {self._IMAGE_FORMATS}. Got: '{image_format}'")

        # Check the shape of input points
        if input_points.ndim != 2 or input_points.shape[1] != 2:
            raise ValueError(
                f"Invalid input points shape. Expected 2D array with shape (N, 2) "
                f"as (x, y) coordinates. Got: {input_points.shape}"
            )
        if len(input_points) == 0:
            raise ValueError("Input points arrays must not be empty")

        # Check the shape of target labels
        if target_labels.ndim != 1:
            raise ValueError(
                f"Invalid target labels shape. Expected 1D array with shape (N,) "
                f"as class indices. Got: {target_labels.shape}"
            )

        # Check that input points and target labels match
        if input_points.shape[0] != target_labels.shape[0]:
            raise ValueError(
                f"Input points and target labels must have the same number of elements. "
                f"Got: input_points={input_points.shape[0]}, target_labels={target_labels.shape[0]}"
            )

        # Check unique target labels with class names
        uq_target_labels = set(target_labels)
        n_classes = len(sorted_classes)
        if len(uq_target_labels) != n_classes:
            logger.warning(
                f"Number of unique target labels ({len(uq_target_labels)}) does not match "
                f"the number of sorted class names ({n_classes})"
            )

        # Check the target labels are within the range of class name indices
        # if uq_target_labels != set(range(n_classes)):
        diff = uq_target_labels - set(range(n_classes))
        if len(diff) > 0:
            raise ValueError(
                f"Target labels must be in the range of class names indices. "
                f"Got: unique_labels={uq_target_labels}. Expected range: [0, {n_classes - 1}]"
            )

        # Check for invalid x and y coordinates
        height, width = image.shape[:2]
        x_pts, y_pts = input_points[:, 0], input_points[:, 1]
        invalid_x_mask = (x_pts < 0) | (x_pts >= width)     # A valid x is in the range [0, width - 1]
        invalid_y_mask = (y_pts < 0) | (y_pts >= height)    # A valid y is in the range [0, height - 1]
        invalid_points_mask = invalid_x_mask | invalid_y_mask
        if numpy.any(invalid_points_mask):
            invalid_points = input_points[invalid_points_mask].tolist()
            raise ValueError(
                f"Input points must be valid pixel coordinates within the image dimensions. "
                f"Image shape: {image.shape}. Invalid points found: {invalid_points}. "
                f"Valid x range: [0, {width - 1}], valid y range: [0, {height - 1}]"
            )

        # Preprocess the input data
        self._preprocess(image, input_points, target_labels, sorted_classes, image_format)

    def predict(
            self,
            include_background: bool,
            verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs SAM predictor for each class.

        Parameters
        ----------
        include_background : bool, optional
            Whether to include background points in the input prompts for each class.
        
        verbose : bool, optional
            Whether to print verbose output during prediction. If True, information is 
            logged using ``logging.INFO`` level. Default is False.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing two elements:

            1.  A tensor of shape (n_classes, height, width) with the predicted logits for
                each class. The logits are un-thresholded values. Classes without input points
                are filled with ``-inf``.

            2.  A tensor of shape (n_classes,) with the SAM predicted mask quality score
                of each class. Classes without input points have a score of 0.
        """
        # The accumulated prediction time for all classes
        final_prediction_time = 0.0

        # Initialize a logit tensor for all classes
        # Fill with a very low value so that unpredicted classes don't win argmax
        mask_logits = torch.full(
            size=( len(self._class_names), self._img_height, self._img_width ),
            fill_value=-float("inf"),
            device=self.device,
            dtype=torch.float32
        )
        iou_scores = torch.zeros(len(self._class_names), device=self.device, dtype=torch.float32)

        # Per-class inference
        for class_name in self._class_names:
            # Input points and labels for the current class
            input_points, input_labels = self._get_class_points(class_name, include_background)
            if (input_points.size == 0) or (input_labels.size == 0):
                logger.warning(f"('{class_name}') No input points, skipping predictions.")
                continue

            try:
                # Log the number of input points for the class
                class_idx = self._class_names.index(class_name)
                if verbose:
                    logger.info(
                        f"('{class_name}') Predicting mask: "
                        f"id={class_idx}, "
                        f"total_points={len(input_points)}, "
                        f"foregrounds={numpy.sum(input_labels == 1)}, "
                        f"backgrounds={numpy.sum(input_labels == 0)}"
                    )

                # Determine whether prompt is ambiguous (single point vs multi-point) 
                is_ambiguous = len(input_points) == 1

                with torch.no_grad():
                    # Make predictions using the SAM predictor
                    #       masks_logits --> array of shape: (n_masks, height, width)
                    #        mask_scores --> array of shape: (n_masks, )
                    #     low_res_logits --> array of shape: (n_masks, 256, 256)
                    t0 = time.perf_counter()
                    masks_logits, mask_scores, low_res_logits = self.predictor.predict(
                        point_coords=input_points,
                        point_labels=input_labels,
                        multimask_output=is_ambiguous,  # Output a single best mask/logit (n_masks=1) for ambiguous prompts, otherwise output 3 masks/logits (n_masks=3)
                        return_logits=True,             # Get un-thresholded masks logits instead of a binary mask
                    )
                    predictions_time = time.perf_counter() - t0

                    # Handle ambiguous vs non-ambiguous prompts
                    # masks_logits new shape: (height, width) 
                    iou_score: float = 0
                    if is_ambiguous: 
                        # Pick the mask with the highest predicted IoU score 
                        best_mask_idx = numpy.argmax(mask_scores) 
                        masks_logits = masks_logits[best_mask_idx]
                        iou_score = float(mask_scores[best_mask_idx])
                    else: 
                        # Take the first mask/logits for non-ambiguous prompts
                        masks_logits = masks_logits[0]
                        iou_score = float(mask_scores[0])

                    # Update the prediction time
                    # self.processing_times["predictions"] = predictions_time
                    self.prediction_times[class_name] = predictions_time
                    final_prediction_time += predictions_time
                    if verbose:
                        logger.info(f"('{class_name}') Prediction time: {predictions_time:.4f} seconds")

                    # Check expected output shapes
                    if masks_logits.shape != (self._img_height, self._img_width):
                        raise ValueError(
                            f"Unexpected predicted masks shape. Got: {masks_logits.shape}. "
                            f"Expected: {(self._img_height, self._img_width)}"
                        )

                # Store the logits and the quality score in the class index slot
                mask_logits[class_idx, :, :] = torch.from_numpy(masks_logits).squeeze()
                iou_scores[class_idx] = iou_score

            except Exception as e:
                raise ValueError(f"('{class_name}') Failed to make SAM predictions: {e}") from e

        # Update the final prediction time
        self.processing_times["predictions"] = final_prediction_time
        if verbose:
            logger.info(f"Total prediction time: {final_prediction_time:.4f} seconds")

        return mask_logits, iou_scores

    def postprocess(
            self,
            class_logits: torch.Tensor,
            iou_scores: torch.Tensor,
            normalize: bool,
            weight_by_scores: bool,
            ignore_label: Optional[int],
            mask_threshold: Optional[float] = None,
            verbose: bool = False
    ) -> Dict[str, NDArray]:
        """
        Post-process the predicted logits and generate the final prediction map.

        Parameters
        ----------
        class_logits : torch.Tensor
            Predicted raw logits for each class with shape (n_classes, height, width).
        
        iou_scores : torch.Tensor
            SAM predicted mask quality score for each class with shape (n_classes, ).
        
        normalize : bool
            Whether to normalize the logits before post-processing using the sigmoid function.
        
        weight_by_scores : bool
            Whether to weight the normalized logits by the IoU scores before post-processing.

        ignore_label : int, optional
            Label assigned to pixels where no class is predicted as foreground (every class logit is
            below ``mask_threshold``). Must be outside ``[0, n_classes - 1]``.
        
        mask_threshold : float, optional
            Raw logit threshold (``foreground = raw_logits > mask_threshold``). It is always applied 
            to the raw logits, never to the sigmoid/IoU-weighted scores, so it is independent of 
            ``normalize`` and ``weight_by_scores``. 
            
            Only used if ``ignore_label`` is not None. If None (default), the default SAM model 
            ``mask_threshold`` value is used (0.0, equivalent to a sigmoid probability of 0.5).
        
        verbose : bool, optional
            Whether to print verbose output during prediction. If True, information is 
            logged using ``logging.INFO`` level. Default is False.
        
        Returns
        -------
        Dict[str, NDArray]
            The post-processed predictions:

            *   "logits" (NDArray): predicted raw logits for each class with
                shape (n_classes, height, width). Classes without input points are filled with ``-inf``.

            *   "labels" (NDArray): class index of each pixel with shape (height, width), obtained by 
                taking the argmax of ``"scores"`` (raw logits if ``normalize=False``) among the classes 
                predicted as foreground. 
                
                Pixels with no foreground class are set to ``ignore_label`` if it is not ``None``, 
                otherwise an arbitrary class index by argmax (usually 0) is assigned.

            *   "scores" (NDArray): if ``normalize=True``, sigmoid probabilities (optionally IoU-weighted) 
                with shape (n_classes, height, width). Omitted if ``normalize=False``.

        """
        # Check arguments
        if mask_threshold is None:
            mask_threshold = self.mask_threshold
        if weight_by_scores and not normalize:
            raise ValueError("'weight_by_scores=True' requires 'normalize=True'")
        if (ignore_label is not None) and (0 <= ignore_label < len(self._class_names)):
            raise ValueError(
                f"'ignore_label' must be outside the class indices range [0, {len(self._class_names) - 1}]. "
                f"Got: {ignore_label}"
            )
        
        # Values compared across classes
        # class_scores --> Shape: (n_classes, H, W)
        class_scores = class_logits.clone() # TODO: Duplicate memory, but avoids in-place operations on the original logits
        if normalize:
            class_scores = torch.sigmoid(class_logits)  # `-inf` maps to 0
            if weight_by_scores:
                class_scores = class_scores * iou_scores.view(-1, 1, 1)

        candidates = class_scores
        if ignore_label is not None:
            # Only classes predicted as foreground compete for the pixel. The threshold is applied on
            # the raw logits (strict `>`, like SAM), so it does not depend on normalization or weighting
            # If a class has a score less than or equal to the threshold, it is ignored
            candidates = class_scores.masked_fill(class_logits <= mask_threshold, -float("inf"))

        # Get labels from the scores
        if verbose:
            logger.info(f"Applying argmax on class scores: threshold={mask_threshold}, ignore_label={ignore_label}")
        max_scores, mask_labels = torch.max(input=candidates, dim=0)   # Shape: (H, W)

        # Pixels where no class is predicted
        unassigned = max_scores == -float("inf")
        n_unassigned = int(unassigned.sum().item())
        if n_unassigned > 0:
            if ignore_label is not None:
                logger.warning(f"{n_unassigned} pixels have no foreground class. Labeled as {ignore_label}")
                mask_labels = mask_labels.masked_fill(unassigned, ignore_label)
            else:
                logger.warning(
                    f"There are {n_unassigned} pixels with no class predicted (max score was -inf). "
                    f"They will be assigned an arbitrary class index by argmax (usually 0)."
                )

        results = {
            "logits": class_logits.cpu().numpy(),
            "labels": mask_labels.cpu().numpy().astype(int),
        }
        if normalize:
            results["scores"] = class_scores.cpu().numpy()

        return results

    def make_inference(
            self,
            include_background: bool,
            normalize: bool = True,
            weight_by_scores: bool = False,
            ignore_label: Optional[int] = None,
            mask_threshold: Optional[float] = None,
            verbose: bool = False
    ) -> NDArray:
        """
        Runs the SAM predictor on the input image and returns the predicted labels.

        SAM is trained for promptable binary segmentation, so the raw logits of the independent
        per-class forward passes (each with a different prompt configuration) are not guaranteed
        to be calibrated or comparable across classes. The resulting ``"labels"`` map is therefore
        a heuristic. The options ``normalize`` and ``weight_by_scores`` can be used to mitigate 
        this issue and improve the comparability of the logits across classes.

        Parameters
        ----------
        include_background : bool
            If True, include background points (negative prompts) in the model input.
            Default is False.

        normalize : bool, optional
            If True (default), the per-class logits are mapped to probabilities with a sigmoid
            before being compared across classes. If False, SamPredictor raw logits are compared directly.

        weight_by_scores : bool, optional
            If True, the sigmoid probabilities of each class are multiplied by the SAM predicted
            mask quality score (IoU prediction) of that class before comparing them.
            Requires ``normalize=True``. Default is False.

        ignore_label : int, optional
            Label assigned to pixels where no class is predicted as foreground (every class logit is
            below ``mask_threshold``). Must be outside ``[0, n_classes - 1]``.
            If None (default), every pixel is assigned to a class by argmax (legacy behavior).

        mask_threshold : float, optional
            Raw logit threshold (``foreground = raw_logits > mask_threshold``). It is always applied 
            to the raw logits, never to the sigmoid/IoU-weighted scores, so it is independent of 
            ``normalize`` and ``weight_by_scores``. 
            
            Only used if ``ignore_label`` is not None. If None (default), the default SAM model 
            ``mask_threshold`` value is used (0.0, equivalent to a sigmoid probability of 0.5).
        
        verbose : bool, optional
            Whether to print verbose output during prediction. If True, information is 
            logged using ``logging.INFO`` level. Default is False.

        Returns
        -------
        NDArray
            Class index of each pixel with shape (height, width), obtained by taking the argmax of 
            ``"scores"`` (raw logits if ``normalize=False``) among the classes predicted as foreground.
            
            Pixels with no foreground class are set to ``ignore_label`` if it is not ``None``, 
            otherwise an arbitrary class index by argmax (usually 0) is assigned.

        """
        # Check if the image embedding has been set
        if not self._img_set:
            raise ValueError("Image embedding has not been set. Use 'set_input()' method first.")

        # Log the inference mode
        if verbose:
            add_txt = "(using background)" if include_background else "(only foreground)"
            logger.info(f"Running per-class predictions {add_txt}")
        
        # Run predictions for each class
        mask_logits, iou_scores = self.predict(include_background, verbose)
        if verbose:
            logger.info("Processing the predictions")

        # Post-process the predictions
        if verbose:
            logger.info("Post-processing the predictions")
        results = self.postprocess(
            mask_logits,
            iou_scores,
            normalize=normalize,
            weight_by_scores=weight_by_scores,
            ignore_label=ignore_label,
            mask_threshold=mask_threshold,
            verbose=verbose
        )

        # Clear GPU memory
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return results["labels"]

    





def plot_class_logit_distributions(
        logits: NDArray,
        class_names: List[str],
        threshold: float = 0.0,
        max_samples: int = 200_000,
        bins: int = 100,
        random_seed: Optional[int] = 0,
        figsize: Tuple[float, float] = (14, 5)
) -> Tuple[Any, Dict[str, Dict[str, float]]]:
    """
    Plot the per-class raw logit distributions to check whether their scales are comparable.

    Two panels are drawn: overlaid density histograms and box plots. If the classes show clearly
    different locations or spreads, a direct argmax across the logits is not reliable.

    Parameters
    ----------
    logits : NDArray
        Raw logits with shape (n_classes, height, width), as in the ``"logits"`` output of
        ``SparsePointsSamPredictor.make_inference``. Non-finite values (e.g. classes skipped
        for lack of input points) are ignored.

    class_names : List[str]
        Class names ordered by class index.

    threshold : float, optional
        Foreground logit threshold, drawn as a vertical line. Default is 0.0 (SAM's ``mask_threshold``).

    max_samples : int, optional
        Maximum number of randomly sampled pixels per class used for the plots and statistics.

    bins : int, optional
        Number of histogram bins.

    random_seed : int, optional
        Seed for the pixel sampling.

    figsize : Tuple[float, float], optional
        Figure size.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, Dict[str, Dict[str, float]]]
        The figure and the per-class summary statistics (``count``, ``mean``, ``std``, ``min``,
        ``p5``, ``median``, ``p95``, ``max`` and ``foreground_fraction``, the fraction of pixels
        with a logit above ``threshold``).
    """
    import matplotlib.pyplot as plt

    if logits.ndim != 3 or logits.shape[0] != len(class_names):
        raise ValueError(
            f"Expected logits with shape (n_classes={len(class_names)}, H, W). Got: {logits.shape}"
        )

    rng = numpy.random.default_rng(random_seed)
    samples: Dict[str, NDArray] = {}
    stats: Dict[str, Dict[str, float]] = {}
    for idx, class_name in enumerate(class_names):
        values = logits[idx].ravel()
        values = values[numpy.isfinite(values)]
        if values.size == 0:
            logger.warning(f"('{class_name}') No finite logits to plot, skipping")
            continue
        if values.size > max_samples:
            values = rng.choice(values, size=max_samples, replace=False)

        samples[class_name] = values
        p5, median, p95 = numpy.percentile(values, [5, 50, 95])
        stats[class_name] = {
            "count": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "p5": float(p5),
            "median": float(median),
            "p95": float(p95),
            "max": float(values.max()),
            "foreground_fraction": float(numpy.mean(values >= threshold)),
        }

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax_hist, ax_box = axes[0], axes[1]
    if samples:
        all_values = numpy.concatenate(list(samples.values()))
        bin_edges = numpy.linspace(all_values.min(), all_values.max(), bins + 1)
        for class_name, values in samples.items():
            ax_hist.hist(values, bins=bin_edges, density=True, alpha=0.4, label=class_name)
        ax_box.boxplot(list(samples.values()), showfliers=False)
        ax_box.set_xticks(range(1, len(samples) + 1))
        ax_box.set_xticklabels(list(samples.keys()), rotation=45)

    for ax in (ax_hist, ax_box):
        axis_line = ax.axvline if ax is ax_hist else ax.axhline
        axis_line(threshold, color="k", linestyle="--", linewidth=1, label=f"threshold = {threshold:g}")
    ax_hist.set_xlabel("Raw logit")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Per-class logit distributions")
    ax_hist.legend()
    ax_box.set_ylabel("Raw logit")
    ax_box.set_title("Per-class logit spread")
    fig.tight_layout()

    return fig, stats
