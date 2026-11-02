import abc

from numpy.typing import ArrayLike, NDArray
from typing import (
    Tuple
)



# TODO: Maybe rename to BaseSplitData with train(), validate(), and test() methods
# class BaseDataset(abc.ABC):
class OptimizationData(abc.ABC):
    """
    Abstract base class for datasets used in machine learning tasks.
    """

    @abc.abstractmethod
    def uses_validation(self) -> bool:
        """Indicates whether the dataset provides a separate validation set."""
        raise NotImplementedError()

    @abc.abstractmethod
    def train(self, **kwargs) -> Tuple[NDArray, NDArray]:
        """
        Retrieve the dataset's features (``X``) and targets (``y``) for training  
        as a tuple with two elements:

        * ``X``: A 2D array of shape (``n_samples``, ``n_features``).
        * ``y``: A 1D or 2D array of shape (``n_samples``,) or (``n_samples``, ``n_targets``). 
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def evaluate(self, **kwargs) -> Tuple[NDArray, NDArray]:
        """
        Retrieve the dataset's features (``X``) and targets (``y``) for validation
        as a tuple with two elements:

        * ``X``: A 2D array of shape (``n_samples``, ``n_features``).
        * ``y``: A 1D or 2D array of shape (``n_samples``,) or (``n_samples``, ``n_targets``).
        """
        raise NotImplementedError()

    # @abc.abstractmethod
    # def test_data(self, **kwargs) -> Tuple[NDArray, ArrayLike]:
    #     """Retrieve the dataset's features (x) and labels (y) for testing as a tuple (x, y)."""
    #     raise NotImplementedError()


# TODO: 
#   * Implement a PandasDataset class based on MLFlow:
#       https://mlflow.org/docs/latest/api_reference/python_api/mlflow.data.html#mlflow.data.pandas_dataset.PandasDataset
#   * Must include data as:
#       + Columns for features (x) and targets (y).
#       + Split column and values for train, validation, and test sets. (maybe think about a Dataset per split)
class PandasDataset(BaseDataset):
    pass