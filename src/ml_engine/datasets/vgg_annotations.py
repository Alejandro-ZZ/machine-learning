"""
Class for handling VGG Image Annotator annotations.

References: 

    * [VIA2 Documentation](https://gitlab.com/vgg/via-suite/via/-/blob/master/via-2.x.y/CodeDoc.md#description-of-via-project-json-file)
    * [VIA2 Utility Scripts](https://gitlab.com/vgg/via-suite/via/-/tree/master/via-2.x.y/scripts?ref_type=heads)

"""
import json
import numpy
import pandas
import shapely
import warnings

from numpy.typing import NDArray
from pathlib import Path
from scipy.spatial import KDTree
from shapely.geometry.base import BaseGeometry
from skimage import draw, io
from typing import Any, Dict, Optional, List, Self, Tuple, Union


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

    def combine_attributes(self) -> dict:
        """
        Combine shape attributes and region attributes into a single dictionary.

        Returns
        -------
        dict
            A dictionary containing both shape and region attributes::
            
                {
                    "name": "rect",
                    "x": 100,
                    "y": 150,
                    "width": 200,
                    "height": 100,

                    "label": "cat",
                    "occluded": false
                }
        """
        combined_attrs = self.shape_attrs.copy()
        combined_attrs.update(self.region_attrs)
        return combined_attrs

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

    # TODO: 
    #   Optionally resample the pixel coordinates based on desired distance to 
    #   reduce the number of points for large regions.
    def pixel_coordinates(self, image_shape: Optional[tuple] = None, distance: Optional[float] = None) -> Tuple[NDArray, ...]:
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
        
        distance : float, optional
            If provided, this parameter specifies the minimum distance between points.

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


        # TODO: Under implementation
        if (distance is not None) and (shape_name not in {"point", "polyline"}):
            points_arr = numpy.column_stack(output_coords)
            tree = KDTree(data=points_arr)
            keep = numpy.ones(len(points_arr), bool)
            result = []
            for idx, coordinate in enumerate(points_arr):
                if keep[idx]:
                    # mark all neighbors within "distance" as “excluded”
                    result.append(coordinate)
                    ids = tree.query_ball_point(coordinate, distance)
                    keep[ids] = False

            result = numpy.array(result)
            output_coords = (result[:, 0], result[:, 1])

        return output_coords


class VggImageAnnotations:
    """
    A class to handle annotations from the VGG Image Annotator (VIA version 2) tool.

    Attributes
    ----------
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
        Information about all images and their associated metadata. Each image is indexed using the filename 
        and contains the following information::

            "_via_img_metadata": {
                "<image_filename>": {
                    "filename": <string>,       <-- Image filename
                    "size": <int>,              <-- Image size in bytes
                    "regions": [ ... ],         <-- List of annotated regions. See ``VggRegion`` for structure details
                    "file_attributes": { ... }, <-- File-level attributes
                    "id": <string>,             <-- VGG image ID: <filename><size>
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

        # Store the base path for the images
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
        self.images_metadata: Dict[str, dict] = self.check_images_metadata(annotations)

    @classmethod
    def from_json(cls, filepath: Union[str, Path], base_path: Union[str, Path]) -> "VggImageAnnotations":
        """
        Create a VggImageAnnotations instance from a JSON file.

        Parameters
        ----------
        filepath : str | Path
            The path to the JSON file containing the annotations.
        
        base_path : str | Path
            The base path to the directory containing the annotated image files.
        """
        filepath = Path(filepath)
        base_path = Path(base_path)
        with filepath.open(mode="r", encoding="utf-8") as f:
            annotations = json.load(f)
        return cls(annotations, base_path, is_project="_via_img_metadata" in annotations)

    def to_json(self, filepath: Union[str, Path]):
        """Save the current annotations to a JSON file."""
        filepath = Path(filepath)
        data_to_save = {
            "_via_settings": self.settings,
            "_via_img_metadata": self.images_metadata,
            "_via_attributes": self.attributes,
            "_via_data_format_version": self.data_version,
            "_via_image_id_list": self.image_ids
        }
        with filepath.open(mode="w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=4)

    def check_images_metadata(self, annotations: dict) -> Dict[str, dict]:
        """
        Check and return the image metadata from the provided annotations.
        
        Notes
        -----
        *   Raw metadata is extracted from the "_via_img_metadata" key if present, 
            otherwise it is assumed to be directly the image metadata.
        
        *   The returned metadata change the structure to be indexed by image filename 
            instead of the original image ID. 
        """
        raw_metadata: Dict[str, dict]
        new_metadata: Dict[str, dict] = {}

        if "_via_img_metadata" in annotations:
                # If the annotations are from a project, we extract the image metadata from the "_via_img_metadata" key.
                raw_metadata = annotations["_via_img_metadata"]
        else:
            # If the annotations are not from a project, we assume they are directly the image metadata.
            raw_metadata = annotations

        # Annotated files sanity check
        for image_id, image_data in raw_metadata.items():
            # Check for required keys
            if "regions" not in image_data:
                raise ValueError(f"Missing 'regions' key in image metadata: {image_data}")
            if "filename" not in image_data:
                raise ValueError(f"Missing 'filename' key in image metadata: {image_data}")

            # Get image annotation data
            image_filename = image_data["filename"]
            image_regions = image_data["regions"]

            # Check if the image file exists in the base path
            image_file = self.base_path / image_filename
            if not image_file.exists():
                raise FileNotFoundError(f"Image file not found. Name: '{image_filename}', Path: '{image_file}'")

            # Warn for empty annotations
            if not image_regions:
                warnings.warn(f"Image '{image_filename}' has no annotations (empty 'regions').")

            # Store the original image ID for reference
            image_data["id"] = image_id

            # Add the image data to the new metadata
            new_metadata[image_filename] = image_data

        return new_metadata

    def merge(self, other: "VggImageAnnotations"):
        """
        Merge another VggImageAnnotations instance into this one.

        Parameters
        ----------
        other : VggImageAnnotations
            Another instance of VggImageAnnotations to merge with this one.
        """
        # Merge the settings, attributes, and data version
        self.settings.update(other.settings)
        self.attributes.update(other.attributes)
        self.data_version = max(self.data_version, other.data_version)

        # Merge the image IDs
        self.image_ids.extend(other.image_ids)

        # Merge the images metadata (include both existing and new images)
        for other_filename, other_metadata in other.images_metadata.items():
            if other_filename in self.images_metadata:
                warnings.warn(
                    f"Image '{other_filename}' already exists in the current annotations. " 
                    f"Merging regions and file attributes."
                )

                # Merge regions for existing images
                image_regions: List[dict] = self.images_metadata[other_filename]["regions"]
                image_regions.extend(other_metadata["regions"])

                # Merge file attributes for existing images
                file_attributes: Dict[str, Any] = self.images_metadata[other_filename]["file_attributes"]
                file_attributes.update(other_metadata["file_attributes"])
            else:
                # If the image is new, add it to the metadata
                self.images_metadata[other_filename] = other_metadata



def vgg_to_dataframe(vgg_annotations: VggImageAnnotations) -> pandas.DataFrame:
    """
    Convert VGG annotations to a pandas DataFrame.
    Each row corresponds to a region in the VGG annotations, with columns: 
    
        * Filename: The name of the image file.
        * VggShape: The shape type of the region (e.g., "rect", "circle", etc.) concatenated with the region ID.
        * Cx: The x-coordinate of the pixel (column index).
        * Cy: The y-coordinate of the pixel (row index).
        * Additional columns are added for each region attribute found across all regions.
    """
    df_rows: List[dict] = []    
    region_attrs = set()

    # Iterate over each image in the annotations
    for image_metadata in vgg_annotations.images_metadata.values():
        
        # Get image shape
        image_file = vgg_annotations.base_path / image_metadata["filename"]
        image_shape = io.imread(image_file).shape
        
        # Iterate over each region in the image
        for region_id, region_data in enumerate(image_metadata["regions"], start=1):
            # Create an instance of VggRegion
            vgg_region = VggRegion(
                shape_attrs=region_data["shape_attributes"], 
                region_attrs=region_data["region_attributes"]
            )

            # Add the region attributes to the set
            region_attrs.update(vgg_region.region_attrs.keys())

            # Get pixel coordinates
            rows, cols = vgg_region.pixel_coordinates(image_shape=image_shape)

            # Create a row for the DataFrame
            for row, col in zip(rows, cols):
                df_row = {
                    "Filename": image_metadata["filename"],
                    "VggShape": vgg_region.shape_attrs["name"] + str(region_id),
                    "Cx": int(col),
                    "Cy": int(row),
                    **vgg_region.region_attrs
                }
                df_rows.append(df_row)

    # Create the DataFrame from the list of rows
    df = pandas.DataFrame(data=df_rows)

    print(f"Region attributes found: {list(region_attrs)}")
    return df


