"""
Class for handling VGG Image Annotator annotations.

References: 

    * [VIA2 Documentation](https://gitlab.com/vgg/via-suite/via/-/blob/master/via-2.x.y/CodeDoc.md#description-of-via-project-json-file)
    * [VIA2 Utility Scripts](https://gitlab.com/vgg/via-suite/via/-/tree/master/via-2.x.y/scripts?ref_type=heads)

"""
import json
import math
import matplotlib.pyplot as plt
import numpy
import pandas
import shapely
import warnings

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from pathlib import Path
from scipy.spatial import KDTree
from shapely.geometry.base import BaseGeometry
from shapely.plotting import plot_polygon, plot_line, plot_points
from skimage import draw, io
from typing import Any, Dict, Optional, List, Self, Tuple, Union


# TODO: Debugging code for testing the VggRegion class
"""
import matplotlib.pyplot as plt
from src.ml_engine.datasets.vgg_annotations import *
from shapely.plotting import plot_polygon, plot_line, plot_points

test_attrs = {
    "rect": {"name": "rect", "x": 20, "y": 30, "width": 50, "height": 100},
    "circle": {"name": "circle", "cx": 50, "cy": 50, "r": 30},
    "ellipse": {"name": "ellipse", "cx": 100, "cy": 100, "rx": 50, "ry": 30, "theta": 45},
    "polygon": {"name": "polygon", "all_points_x": [10, 20, 30], "all_points_y": [10, 20, 10]},
    "point": {"name": "point", "cx": 50, "cy": 50},
    "polyline": {"name": "polyline", "all_points_x": [10, 20, 30], "all_points_y": [10, 20, 10]}
}

def test_region(region, distance=10):
    geometry = region.geometry()
    coords = region.pixel_coordinates(distance=None)
    coords2 = region.pixel_coordinates(distance=distance)
    #
    if geometry.geom_type == "Polygon":
        plot_polygon(region.geometry(), color="green", label="Geometry")
    elif geometry.geom_type == "LineString":
        plot_line(region.geometry(), color="green", label="Geometry")
    elif geometry.geom_type == "Point":
        plot_points(region.geometry(), color="green", label="Geometry")
    plt.scatter(coords[1], coords[0], color="blue", label="All Pixels")
    plt.scatter(coords2[1], coords2[0], color="red", label=f"Sparsed (d={distance})")
    #
    plt.legend()
    plt.title(str(region.shape_attrs))
    # plt.xlim(0, 200); plt.ylim(0, 200)
    plt.show()

# rect, circle, ellipse, polygon, point, polyline
region = VggRegion(shape_attrs=test_attrs["rect"], region_attrs={})
# region.pixel_coordinates(distance=3)
test_region(region, distance=3)

"""

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

        Notes
        -----
        Geometries can be plotted with matplotlib using the ``shapely.plotting`` module::
            
            import matplotlib.pyplot as plt
            from shapely.plotting import plot_polygon, plot_line, plot_points

            region = VggRegion(shape_attrs=shape_attrs, region_attrs=region_attrs)
            geom = region.geometry()

            if geom.geom_type == "Polygon":
                plot_polygon(geom)
            elif geom.geom_type == "LineString":
                plot_line(geom)
            elif geom.geom_type == "Point":
                plot_points(geom)

            plt.show()

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

    def pixel_coordinates(
            self, 
            image_shape: Optional[tuple] = None, 
            distance: Optional[float] = None
    ) -> Tuple[NDArray, ...]:
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
            If provided, the pixel coordinates will be sparsified such that no two pixels are 
            closer than this distance. This is useful for reducing the number of pixels in 
            large regions while maintaining coverage.

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
            rr, cc = draw.rectangle(start=(y, x), extent=(height, width), shape=image_shape)
            output_coords = (rr.ravel(), cc.ravel())

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
            theta = math.radians(self.shape_attrs.get("theta", 0))
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


        # Sparsify the pixel coordinates
        if (distance is not None) and (shape_name != "point"):
            points_arr = numpy.column_stack(output_coords)
            tree = KDTree(data=points_arr)  # Expected data shape: (n_pixels, 2) for 2D coordinates
            remaining = set(range(len(points_arr)))
            result = []
            while remaining:
                point_idx = remaining.pop()
                coordinate = points_arr[point_idx]
                result.append(coordinate)
                ids = tree.query_ball_point(coordinate, distance)
                remaining.difference_update(ids)
            result = numpy.array(result)
            output_coords = (result[:, 0], result[:, 1])

        return output_coords


# TODO: 
#   * Try to get the ``base_path`` from the ``_via_settings`` attribute if not provided at annotations["_via_settings"]["core"]["default_filepath"].
#   * Add support for multiple search filepaths. Similar to data at annotations["_via_settings"]["core"]["filepath"] which is a mapping of search filepath to integer ID.
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

        # Mapping of image filenames to their associated metadata, including regions and file attributes
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
        # We need to match the structure of a VIA project JSON file 
        # for the modified ``images_metadata`` attribute.
        # There are two original attribute changes:
        #   1. The image metadata is now indexed by filename instead of the original image ID.
        #   2. The original image ID is stored in the "id" key of each image's metadata for reference.
        # Finally if no image ids are present, we will generate them from the "id" key in the image metadata.
        original_images_metadata: Dict[str, dict] = {}
        for filename, image_data in self.images_metadata.items():
            # Store the original metadata structure
            image_id = image_data["id"]
            original_images_metadata[image_id] = image_data
            original_images_metadata[image_id].pop("id")

            # If the image ID is not already in the list, add it
            if image_id not in self.image_ids:
                self.image_ids.append(image_id)

        filepath = Path(filepath)
        data_to_save = {
            "_via_settings": self.settings,
            "_via_img_metadata": original_images_metadata,
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

            # Store the image ID if it is not already in the list 
            # (occurs when creating a new instance from a non-project JSON file)
            if image_id not in self.image_ids:
                self.image_ids.append(image_id)

            # Add the image data to the new metadata
            new_metadata[image_filename] = image_data

        return new_metadata

    def merge(self, other: "VggImageAnnotations"):
        """
        Merge another VggImageAnnotations instance into this one.

        Notes
        -----
        *   The ``settings`` and ``attributes`` attributes are updated with the 
            values from the other instance. If there are conflicting keys, the 
            values from the other instance will overwrite the current ones.

        *   The ``data_version`` is updated to the maximum of the two instances.

        *   The ``image_ids`` list is extended with the image IDs from the other 
            instance, ensuring that there are no duplicates. If an image ID already 
            exists, a warning is issued.
        
        *   For existing images in the ``images_metadata``, the "regions" list is 
            extended with the regions from the other instance, and the "file_attributes" 
            dictionary is updated.

        *   For new images in the ``images_metadata``, the image data from the other 
            instance is added directly.

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
        # self.image_ids.extend(other.image_ids)
        for other_image_id in other.image_ids:
            if other_image_id not in self.image_ids:
                self.image_ids.append(other_image_id)
            else:
                warnings.warn(f"Image ID '{other_image_id}' already exists in the current annotations. Skipping.")

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


    def draw_regions(
            self, 
            image_filename: str, 
            *,
            color: Optional[Any] = None,
            color_attribute: Optional[str] = None,
            legend: bool = True,
            ax: Optional[Axes] = None
    ) -> Axes:
        """
        Draw the annotated regions of a specific image on a matplotlib Axes.

        Notes
        -----
        *   When generating axes legend, it can be overlapping or cutted if using 
            ``plt.tight_layout()``. It is recommented to use ``plt.subplots(layout="constrained")`` 
            or ``plt.savefig("my_plot.png", bbox_inches="tight")`` if saving the figure to a file. 

        Examples
        --------
        * Draw regions with default colors and legend::

            annotaations = VggImageAnnotations.from_json("annotations.json", base_path="images/")
            annotations.draw_regions("image.jpg")

            
        * Generate a figure for each annotated image in the annotations::

            for image_filename in annotations.images_metadata.keys():
                fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
                annotations.draw_regions(image_filename, color_attribute="class", ax=ax)
                ax.set_title(f"{image_filename}", fontweight="bold")
                ax.axis("off")
                fig.savefig(f"{image_filename}_annotations.png", dpi=300)
                plt.show()
                plt.close(fig)

        Parameters
        ----------
        image_filename : str
            The filename of the image whose regions are to be drawn.
        
        color : Any, optional
            Fixed color to use for all regions. If None (default), the matplotlib 
            default color cycle will be used. Ignored if ``color_attribute`` is provided.

            Data type must be compatible with matplotlib color specifications, e.g., a 
            string (like "red" or "#FF0000"), a tuple of RGB or RGBA values, or a matplotlib 
            colormap.

        color_attribute : str, optional
            If provided, regions will be colored based on the unique values of this 
            attribute in the region attributes. 
            
            Each unique value will be assigned a different color. If a region does not 
            have this attribute, it will be assigned a default color.

            Color palette used is the "tab10" colormap from matplotlib. If there are more 
            than 10 unique values, the colors will cycle through the colormap. 

        legend : bool, optional
            If True (default), a legend will be added to the plot indicating the mapping 
            between the unique values of the ``color_attribute`` and their corresponding colors.
            Ignored if ``color_attribute`` is None.
        
        ax : matplotlib.axes.Axes, optional
            The Axes on which to draw the regions. If None, a new figure and axes will be created.
        
        Returns
        -------
        matplotlib.axes.Axes
            The Axes object with the drawn regions.
        """
        # Check if the image exists in the metadata
        if image_filename not in self.images_metadata:
            raise ValueError(f"Image '{image_filename}' not found in annotations.")

        # Load the image
        image_path = self.base_path / image_filename
        try:
            image = io.imread(image_path)
        except Exception as e:
            raise ValueError(f"Failed to load image: filename='{image_filename}', path='{image_path}'. Error: {e}")

        # Create a new figure and axes if none provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")

        # Color mapping based on a region attribute
        # Populated dynamically based on the ``color_attribute`` unique values
        pyplot_cmap = plt.get_cmap("tab10") # "tab20"
        color_mapping: Dict[str, Tuple[float, float, float, float]] = {}

        # Display the image
        ax.imshow(image)
        ax.set_title(f"Annotatios for {image_filename}")
        ax.axis("off")

        # Draw each region on the axes
        for region_data in self.images_metadata[image_filename]["regions"]:
            vgg_region = VggRegion(
                shape_attrs=region_data["shape_attributes"],
                region_attrs=region_data["region_attributes"]
            )

            # Determine the color for the region
            if color_attribute:
                attr_value = vgg_region.region_attrs.get(color_attribute, "Unknown")
                if attr_value not in color_mapping:
                    idx = len(color_mapping) % pyplot_cmap.N  # Cycle through the colormap
                    color_mapping[attr_value] = pyplot_cmap(idx)
                color_value = color_mapping[attr_value]
            else:
                color_value = color

            # Draw the region gometry
            geom = vgg_region.geometry()
            if geom.geom_type == "Polygon":
                plot_polygon(polygon=geom, ax=ax, color=color_value)
            elif geom.geom_type == "LineString":
                plot_line(line=geom, ax=ax, color=color_value)
            elif geom.geom_type == "Point":
                plot_points(geom=geom, ax=ax, color=color_value)

        # Add horizontal legend for color mapping    
        if color_attribute and legend:
            line_kwargs: Dict[str, Any] = dict(lw=4, marker="s", color="w", markersize=10)
            handles = [
                Line2D([0], [0], label=attr_val, color=attr_color, markerfacecolor=attr_color, **line_kwargs)
                for attr_val, attr_color in color_mapping.items()
            ]
            ax.legend(
                handles=handles, 
                title=color_attribute.capitalize() , 
                loc="upper center", 
                bbox_to_anchor=(0.5, -0.05), 
                ncol=len(color_mapping)
            )

        return ax






def vgg_to_dataframe(vgg_annotations: VggImageAnnotations) -> pandas.DataFrame:
    """
    Convert VGG annotations to a pandas DataFrame.
    Each row corresponds to a region in the VGG annotations, with columns: 
    
        *   Filename: The name of the image file.

        *   VggShape: The shape type of the region (e.g., "rect", "circle", etc.) concatenated with the region ID.
            Region ID is a unique autoincremented integer for each region in the image, starting from 1.
         
        *   Cx: The x-coordinate of the pixel (column index).
        
        *   Cy: The y-coordinate of the pixel (row index).
        
        *   Additional columns are added for each region attribute found across all regions.
    """
    # List to hold all rows for the DataFrame
    df_rows: List[dict] = []   

    # Set to keep track of all unique region attributes across all images 
    # region_attrs = set()

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
            # region_attrs.update(vgg_region.region_attrs.keys())

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

    # Get all unique region attributes across all regions with its unique values
    region_attrs: Dict[str, set] = {
        attr: set(df[attr]) 
        for attr in df.columns 
        if attr not in {"Filename", "VggShape", "Cx", "Cy"}
    }

    # Display found region attributes
    print(f"Region attributes found:")
    for attr, values in region_attrs.items():
        print(f"  * {attr}: {list(values)}")
    
    return df
