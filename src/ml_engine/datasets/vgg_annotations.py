"""
Class for handling VGG Image Annotator annotations.

References: 

    * [VIA2 Documentation](https://gitlab.com/vgg/via-suite/via/-/blob/master/via-2.x.y/CodeDoc.md#description-of-via-project-json-file)
    * [VIA2 Utility Scripts](https://gitlab.com/vgg/via-suite/via/-/tree/master/via-2.x.y/scripts?ref_type=heads)

"""
import warnings

from pathlib import Path
from typing import Any, Dict, Optional, List


class VggRegion:
    """
    A class to represent a single annotated region in an image, as defined by the VGG Image Annotator (VIA) tool.
    The VGG format for a region is as::

        region = {
            "shape_attributes": {
                "name": "rect",
                "x": 100,
                "y": 150,
                "width": 200,
                "height": 100
            },

            "region_attributes": {
                "label": "cat",
                "occluded": false
            }
        }

    Parameters
    ----------
    shape : dict
        Shape attributes of the region. It must contain a "name" key indicating the type of shape.
        Common shapes include "rect", "circle", "ellipse", "polygon", "point", and "polyline".
    
    attrs : dict
        Additional region attributes associated, such as labels or other metadata.
    """
    # Mapping of shape names to their required attributes
    SHAPE_ATTRIBUTES = {
        "rect": {"x", "y", "width", "height"},
        "circle": {"cx", "cy", "r"},
        "ellipse": {"cx", "cy", "rx", "ry", "theta"},
        "polygon": {"all_points_x", "all_points_y"},
        "point": {"cx", "cy"},
        "polyline": {"all_points_x", "all_points_y"}
    }

    def __init__(self, shape: dict, attrs: dict):
        self.shape = self.validate_shape(shape)
        self.attributes = attrs

    @staticmethod
    def validate_shape(shape: dict) -> dict:
        """Check and return a validated shape attributes of a region."""
        # Name of shape (e.g., "rect", "circle", etc.)
        shape_name = shape.get("name")
        if not shape_name:
            raise ValueError(f"Missing 'name' key in shape: {shape}")

        # Valid shape name
        if shape_name not in VggRegion.SHAPE_ATTRIBUTES:
            raise ValueError(f"Unknown shape type '{shape_name}' in shape attributes: {shape}")

        # Required attributes for the shape
        required_attrs = VggRegion.SHAPE_ATTRIBUTES[shape_name]
        
        # Check for any missing required attributes in the provided shape attributes
        missing_attrs = required_attrs - shape.keys()
        if missing_attrs:
            raise ValueError(f"Missing required attributes: {list(missing_attrs)}. Shape attributes: {shape}")

        return shape



class VggImageAnnotations:
    """
    A class to handle annotations from the VGG Image Annotator (VIA version 2) tool.
    """
    # Required keys for a VIA project JSON file
    PROJECT_KEYS = { 
        "_via_settings", 
        "_via_img_metadata", 
        "_via_attributes", 
        "_via_data_format_version", 
        "_via_image_id_list" 
    }

    def __init__(self, annotations: dict, base_path: Path, is_project: bool = False):
        if not base_path.is_dir():
            raise ValueError(f"Base path must be an already existing valid directory: '{base_path}'")
        if is_project:
            # Ensure that the provided annotations contain all required project keys
            missing_keys = self.PROJECT_KEYS - annotations.keys()
            if missing_keys:
                raise ValueError(f"Missing project keys in annotations: {missing_keys}. Got: {list(annotations.keys())}")

        # Store the raw annotations and the base path for the images
        self.raw_annotations = annotations
        self.base_path = base_path

        # Settings used by the VIA application
        self.settings: Dict[str, Any] = annotations.get("_via_settings", {})

        # Attributes that can be attached to image and its regions
        # This defines the types of attributes that can be associated with images and their regions (annotations).
        self.attributes: Dict[str, Any] = annotations.get("_via_attributes", {})

        # Data format version of the annotations
        self.data_version: str = annotations.get("_via_data_format_version", "0.0.0")

        # List of image IDs present in the annotations
        self.image_ids: List[str] = annotations.get("_via_image_id_list", [])

        # Information about all images and their associated metadata
        # Each image is indexed using a unique key: FILENAME-FILESIZE and contains the following information:
        #   * "filename": The name of the image file.
        #   * "size": The size of the image file in bytes.
        #   * "regions": A list of regions (annotations) associated with the image.
        #   * "file_attributes": A dictionary of additional attributes related to the image file.
        self.images_metadata: Dict[str, dict] = self.check_images_metadata(annotations)


    def check_images_metadata(self, annotations: dict) -> Dict[str, dict]:
        images_metadata: Dict[str, dict]
        if "_via_img_metadata" in annotations:
                # If the annotations are from a project, we extract the image metadata from the "_via_img_metadata" key.
                images_metadata = annotations["_via_img_metadata"]
        else:
            # If the annotations are not from a project, we assume they are directly the image metadata.
            images_metadata = annotations

        # Annotated files sanity check
        for image_data in images_metadata.values():
            # Check for required keys
            if "regions" not in image_data:
                raise ValueError(f"Missing 'regions' key in image metadata: {image_data}")
            if "filename" not in image_data:
                raise ValueError(f"Missing 'filename' key in image metadata: {image_data}")

            # Get image annotation data
            image_name = image_data["filename"]
            image_regions = image_data["regions"]

            # Check if the image file exists in the base path
            image_file = self.base_path / image_name
            if not image_file.exists():
                raise FileNotFoundError(f"Image file not found. Name: '{image_name}', Path: '{image_file}'")

            # Warn for empty annotations
            if not image_regions:
                warnings.warn(f"Image '{image_name}' has no annotations (empty 'regions').")
            
        return images_metadata