"""
Class for handling VGG Image Annotator annotations.

References: 

    * [VIA2 Documentation](https://gitlab.com/vgg/via-suite/via/-/blob/master/via-2.x.y/CodeDoc.md#description-of-via-project-json-file)
    * [VIA2 Utility Scripts](https://gitlab.com/vgg/via-suite/via/-/tree/master/via-2.x.y/scripts?ref_type=heads)

"""
import numpy
import shapely
import warnings

from numpy.typing import NDArray
from pathlib import Path
from shapely.geometry.base import BaseGeometry
from skimage import draw
from typing import Any, Dict, Optional, List, Tuple


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
    shape_attrs : dict
        Shape attributes of the region. It must contain a "name" key indicating the type of shape.
        Common shapes include "rect", "circle", "ellipse", "polygon", "point", and "polyline".
    
    region_attrs : dict
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

    def __init__(self, shape_attrs: dict, region_attrs: dict):
        self.shape_attrs = self.validate_attributes(shape_attrs)
        self.region_attrs = region_attrs

    @staticmethod
    def validate_attributes(shape_attributes: dict) -> dict:
        """Check and return a validated shape attributes of a region."""
        # Name of shape (e.g., "rect", "circle", etc.)
        shape_name = shape_attributes.get("name")
        if not shape_name:
            raise ValueError(f"Missing 'name' key in shape. Got: {shape_attributes}")

        # Valid shape name
        if shape_name not in VggRegion.SHAPE_ATTRIBUTES:
            raise ValueError(
                f"Unsupported shape name '{shape_name}'. "
                f"Possible values: {list(VggRegion.SHAPE_ATTRIBUTES.keys())}"
            )

        # Required attributes for the shape
        required_attrs = VggRegion.SHAPE_ATTRIBUTES[shape_name]
        
        # Check for any missing required attributes in the provided shape attributes
        missing_attrs = required_attrs - shape_attributes.keys()
        if missing_attrs:
            raise ValueError(f"Missing required attributes: {list(missing_attrs)}. Shape attributes: {shape_attributes}")

        return shape_attributes
    
    def geometry(self) -> BaseGeometry:
        """
        Create a Shapely geometric object based on the shape attributes.

        Returns
        -------
        shapely.geometry.base.BaseGeometry
            A Shapely geometric object corresponding to the specified shape.
        """
        # Created shape
        geometry: BaseGeometry

        # Get the shape name
        shape_name: str = self.shape_attrs["name"]

        if shape_name == "rect":
            x = self.shape_attrs["x"]
            y = self.shape_attrs["y"]
            width = self.shape_attrs["width"]
            height = self.shape_attrs["height"]
            geometry = shapely.geometry.box(x, y, x + width, y + height)

        elif shape_name == "circle":
            cx = self.shape_attrs["cx"]
            cy = self.shape_attrs["cy"]
            r = self.shape_attrs["r"]
            geometry = shapely.geometry.Point(cx, cy).buffer(r)

        # TODO: Check theta rotation for ellipse foor compatibility with VGG annotations
        elif shape_name == "ellipse":
            cx = self.shape_attrs["cx"]
            cy = self.shape_attrs["cy"]
            rx = self.shape_attrs["rx"]
            ry = self.shape_attrs["ry"]
            theta = self.shape_attrs.get("theta", 0)
            circle = shapely.geometry.Point(cx, cy).buffer(1)
            ellipse = shapely.affinity.scale(circle, rx, ry)
            geometry = shapely.affinity.rotate(ellipse, theta)

        elif shape_name == "point":
            cx = self.shape_attrs["cx"]
            cy = self.shape_attrs["cy"]
            geometry = shapely.geometry.Point(cx, cy)

        elif shape_name == "polygon":
            all_points_x = self.shape_attrs["all_points_x"]
            all_points_y = self.shape_attrs["all_points_y"]
            geometry = shapely.geometry.Polygon(zip(all_points_x, all_points_y))

        elif shape_name == "polyline":
            all_points_x = self.shape_attrs["all_points_x"]
            all_points_y = self.shape_attrs["all_points_y"]
            geometry = shapely.geometry.LineString(zip(all_points_x, all_points_y))

        else:
            raise ValueError(f"Unknown shape type '{shape_name}' in VGG annotations.")

        return geometry

    def pixel_coordinates(self, image_shape: Optional[tuple]) -> Tuple[NDArray, ...]:
        """
        Generate coordinates of all pixels contained within the region, based on its shape.
        Output coordinates are in (row, col) format for NumPy indexing::

            rows, cols = region.pixel_coordinates(image_shape)
            image[rows, cols] = 1  # Set pixels inside the region to 1

        Parameters
        ----------
        image_shape : tuple, optional
            The shape of the image (height, width). This is used to ensure that the generated 
            pixel coordinates do not exceed the image boundaries.

        Returns
        -------
        Tuple[NDArray, ...]
            A tuple of NumPy arrays representing the row and column indices of the pixels 
            contained within the region. The first array corresponds to row indices, and 
            the second array corresponds to column indices.
        """
        # Get the shape name
        shape_name: str = self.shape_attrs["name"]

        # Coordinates of pixels as (row, col) format for NumPy indexing
        output_coords: Tuple[NDArray, ...]

        if shape_name == "rect":
            x = self.shape_attrs["x"]
            y = self.shape_attrs["y"]
            width = self.shape_attrs["width"]
            height = self.shape_attrs["height"]
            output_coords = draw.rectangle(start=(y, x), extent=(height, width), shape=image_shape)

        elif shape_name == "circle":
            cx = self.shape_attrs["cx"]
            cy = self.shape_attrs["cy"]
            r = self.shape_attrs["r"]
            output_coords = draw.disk(center=(cy, cx), radius=r, shape=image_shape)

        elif shape_name == "ellipse":
            cx = self.shape_attrs["cx"]
            cy = self.shape_attrs["cy"]
            rx = self.shape_attrs["rx"]
            ry = self.shape_attrs["ry"]
            theta = self.shape_attrs.get("theta", 0)
            output_coords = draw.ellipse(r=cx, c=cy, r_radius=rx, c_radius=ry, rotation=theta, shape=image_shape)

        elif shape_name == "point":
            cx = self.shape_attrs["cx"]
            cy = self.shape_attrs["cy"]
            output_coords = draw.disk(center=(cy, cx), radius=1, shape=image_shape)

        elif shape_name == "polygon":
            all_points_x = self.shape_attrs["all_points_x"]
            all_points_y = self.shape_attrs["all_points_y"]
            output_coords = draw.polygon(r=all_points_y, c=all_points_x, shape=image_shape)

        elif shape_name == "polyline":
            all_points_x = self.shape_attrs["all_points_x"]
            all_points_y = self.shape_attrs["all_points_y"]
            rr_list: List[NDArray] = []
            cc_list: List[NDArray] = []
            for i in range(len(all_points_x) - 1):
                r0, c0 = all_points_y[i], all_points_x[i]
                r1, c1 = all_points_y[i + 1], all_points_x[i + 1]
                rr, cc = draw.line(r0=r0, c0=c0, r1=r1, c1=c1)
                rr_list.append(rr)
                cc_list.append(cc)
            output_coords = (numpy.concatenate(rr_list), numpy.concatenate(cc_list))
                
        else:
            raise ValueError(f"Unknown shape type '{shape_name}' in VGG annotations.")

        return output_coords


class VggImageAnnotations:
    """
    A class to handle annotations from the VGG Image Annotator (VIA version 2) tool.

    Attributes
    ----------
    raw_annotations : dict
        The original annotations exported from the VIA JSON file.

    base_path : Path
        The base path to the directory containing the annotated files.
    
    settings : dict
        Settings used by the VIA application, extracted from the annotations::
        
            "_via_settings": {
                "ui": { ... },

                "core": {
                    filepath: { ... },          <-- Mapping of search filepath to integer ID
                    default_filepath: <string>, <-- Default search filepath
                },
                
                "project": { "name": <string> }
            }    
    
    attributes : dict
        Attributes that can be attached to image ("file") and its regions ("region"). Each attribute is 
        defined as a dictionary::
        
            "_via_attributes": {
                "region": {
                    <attribute_name>: {
                        "type": <string>,           <-- "text", "checkbox", "radio", "image", "dropdown"
                        "description": <string>,
                        
                        "options": { ... },         <-- Mapping of option ID and description (omitted for "text" type)
                        "default_options": { ... }, <-- Mapping of default option ID and description (omitted for "text" type)
    
                        "default_value": <string>,  <-- Default value for "text" type attributes only
                    },
                    ...
                },

                "file": { ... }     <-- Same structure as "region" attributes, but for file-level attributes
            }
        
    data_version : str
        Data format version of the annotations, e.g., "2.0.0".
        
    image_ids : List[str]
        A list of image IDs present in the annotations.
    
    images_metadata : Dict[str, dict]
        Information about all images and their associated metadata. Each image is indexed using a unique key: 
        FILENAME-FILESIZE and contains the following information::
        
            "_via_img_metadata": {
                "<image_id>": {
                    "filename": <string>,
                    "size": <int>,              <-- Image size in bytes
                    "regions": [ ... ],         <-- List of annotated regions. See ``VggRegion`` for structure details
                    "file_attributes": { ... }
                },
                ...
            }
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
        """Check and return the image metadata from the provided annotations."""
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




