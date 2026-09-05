"""
Custom SAM predictor for multi-class sparse point inputs. 
"""
import logging
import numpy
import time
import torch

from numpy.typing import NDArray
from typing import (
    Any,
    Dict,
    List,
    Optional
)


logger = logging.getLogger("SAM")


class SparsePointsSamPredictor:

    _SUPPORTED_DEVICES = {"cpu", "cuda"}
    _IMAGE_FORMATS = {"RGB", "BGR"}

    def __init__(self, predictor: Any, device: str = "cpu"):
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
        self.mask_logits: Optional[torch.Tensor] = None     # Mask logits predicted. Shape: (n_classes, H, W)
        self.class_to_points: Dict[str, NDArray] = {}       # Mapping from class names to (x, y) coordinates

        # Private attributes
        # ---------------------------------------------------
        self._class_names: List[str] = []       # Class names for the points
        self._image: Optional[NDArray] = None   # Image to predict on

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"predictor={self.predictor.__class__.__name__}, "
            f"device={self.device}"
            ")"
        )

    def __repr__(self):
        return str(self)


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

    def make_inference(self, use_background_prompts: bool = False):
        """
        Runs the SAM predictor on the input image and returns the predicted logits and labels.

        Parameters
        ----------
        use_background_prompts : bool, optional
            If True, include background points (negative points) in the input prompts.
            Default is False.

        Returns
        -------
        Dict[str, NDArray]
            Dictionary containing the predictions:

            *   "logits" (NDArray): predicted raw logits for each class with
                shape (n_classes, height, width).

            *   "labels" (NDArray): post-processed logits for each pixel with shape (height, width).
                This is obtained by taking the argmax over the raw logits along the class dimension.
        """
        # Check if the image embedding has been set
        if self._image is None:
            raise ValueError("Image embedding has not been set. Use 'set_input' method first.")

        # Run predictions for each class
        if use_background_prompts:
            logger.info("Running per-class predictions using background input prompts")
        else:
            logger.info("Running per-class predictions")
        self._predict(use_background_prompts)

        # Post-process the predictions
        logger.info("Post-processing the predictions")
        results = self._postprocess()

        # Clear GPU memory
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return results


    # Private methods
    # --------------------------------------------------------------
    def _preprocess(
            self,
            image: NDArray,
            input_points: NDArray,
            target_labels: NDArray,
            sorted_classes: List[str],
            image_format: str
    ):
        # Update input attributes
        self._class_names = sorted_classes
        self.class_to_points = {}
        self._image = None

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
        start_time = time.perf_counter()
        try:
            logger.info("Calculating the image embeddings")
            self.predictor.set_image(image=image, image_format=image_format)
            self._image = image
        except Exception as e:
            if self.device == "cuda":
                torch.cuda.empty_cache()
            raise ValueError(f"Fail to calculate image embedding: {e}.") from e
        finally:
            embeddings_time = time.perf_counter() - start_time
        self.processing_times["embeddings"] = embeddings_time
        logger.info(f"Image embeddings calculated in {embeddings_time:.3f} seconds")

        # Initialize a logit tensor for all classes
        # Fill with a very low value so that unpredicted classes don't win argmax
        height, width, _ = self._image.shape
        self.mask_logits = torch.full(
            size=(len(self._class_names), height, width),
            fill_value=-float("inf"),
            device=self.device,
            dtype=torch.float32
        )

    def _predict(self, use_background_prompts: bool = False):
        """
        Runs SAM predictor for each class present in the image's training dataframe subset.

        Returns
        -------
        tuple
            Two elements tuple:

            1. Dictionary containing the below keys. This might be None if any failure occur:

            - "proba": tensor with class probabilities of shape (n_classes, height, width).
            - "labels": tensor with argmax predictions of shape (height, width).
            - "class_points": dictionary mapping class names to their corresponding selected
            coordinates as `numpy` arrays of shape (N, 2) with (x, y) coordinates.

            2. Boolean flag to know if the process was successful.
        """
        # The accumulated prediction time for all classes
        final_prediction_time = 0.0

        # Per-class inference
        for positive_class in self._class_names:
            if positive_class not in self.class_to_points:
                logger.warning(f"No input points for class '{positive_class}'. Skipping predictions.")
                continue

            # Get positive class data
            class_idx = self._class_names.index(positive_class)
            positive_points = self.class_to_points[positive_class]
            logger.debug(f"Predicting '{positive_class}' class (id={class_idx})")

            # Create the input points and labels
            input_points_list = []
            input_labels_list = []
            input_points_list.append(positive_points)
            input_labels_list.append(numpy.ones(shape=len(positive_points), dtype=int))

            # Include negative points in the input points prompt
            if use_background_prompts:
                negative_classes = set(self._class_names) - {positive_class}
                negative_list = [
                    self.class_to_points[negative_class]
                    for negative_class in negative_classes
                    if negative_class in self.class_to_points
                ]
                negative_points = numpy.vstack(negative_list)
                if len(negative_points) > 0:
                    input_points_list.append(negative_points)
                    input_labels_list.append(numpy.zeros(shape=len(negative_points), dtype=int))
                    logger.debug(f"Total background points for class '{positive_class}': {len(negative_points)}")
                else:
                    logger.warning(f"No background points found for class '{positive_class}'")

            # Create the inputs as arrays
            input_points = numpy.vstack(input_points_list)
            input_labels = numpy.concatenate(input_labels_list)

            logger.debug(f"Running SAM predictions with {len(input_points)} input prompts")
            try:
                with torch.no_grad():
                    #   masks_logits --> array of shape: (n_masks, height, width)
                    #         scores --> array of shape: (n_masks, )
                    # low_res_logits --> array of shape: (n_masks, 256, 256)
                    start_time = time.perf_counter()
                    masks_logits, scores, low_res_logits = self.predictor.predict(
                        point_coords=input_points,
                        point_labels=input_labels,
                        return_logits=True,         # Get un-thresholded masks logits instead of a binary mask
                        multimask_output=False,     # Output a single best mask/logit (n_masks=1)
                    )
                    predictions_time = time.perf_counter() - start_time
                    # self.processing_times["predictions"] = predictions_time
                    self.prediction_times[positive_class] = predictions_time
                    final_prediction_time += predictions_time
                    logger.debug(f"Class '{positive_class}' prediction time: {predictions_time:.3f} seconds")

                    # Check expected output shapes
                    expected_masks_shape = (1, self._image.shape[0], self._image.shape[1])
                    if masks_logits.shape != expected_masks_shape:
                        raise ValueError(
                            f"Unexpected predicted masks shape. Expected: {expected_masks_shape}. "
                            f"Predicted: {masks_logits.shape}"
                        )

                # Place the logits in the class index slot
                self.mask_logits[class_idx, :, :] = torch.from_numpy(masks_logits).squeeze()

            except Exception as e:
                raise ValueError(f"Fail to run SAM predictions for class '{positive_class}': {e}") from e

        # Update the final prediction time
        self.processing_times["predictions"] = final_prediction_time
        logger.info(f"Image predictions calculated in {final_prediction_time:.3f} seconds")

    def _postprocess(self) -> Dict[str, NDArray]:
        # Check if argmax resulted in -inf (i.e., no class positively predicted)
        logit_max_values, logit_max_indices = torch.max(input=self.mask_logits, dim=0)
        n_unpredicted = torch.sum(logit_max_values == -float("inf")).item()
        if n_unpredicted > 0:
            logger.warning(
                f"There are {n_unpredicted} pixels with no class predicted (max logit was -inf)."
                f"They will be assigned an arbitrary class index by argmax (usually 0)."
            )

        # Get labels from predicted logits
        logger.debug("Generating prediction map using argmax on aligned logits")
        mask_labels = torch.argmax(input=self.mask_logits, dim=0)   # Shape: (H, W)

        return {
            "logits": self.mask_logits.cpu().numpy(),
            "labels": mask_labels.cpu().numpy().astype(int)
        }
