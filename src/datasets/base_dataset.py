import abc

from numpy.typing import ArrayLike, NDArray
from typing import (
    Tuple
)


class BaseDataset(abc.ABC):
    """
    Abstract base class for datasets used in machine learning tasks.
    """

    @abc.abstractmethod
    def uses_validation(self) -> bool:
        """Indicates whether the dataset provides a separate validation set."""
        raise NotImplementedError()

    @abc.abstractmethod
    def train_data(self, **kwargs) -> Tuple[NDArray, ArrayLike]:
        """Retrieve the dataset's features (x) and labels (y) for training as a tuple (x, y)."""
        raise NotImplementedError()

    @abc.abstractmethod
    def validate_data(self, **kwargs) -> Tuple[NDArray, ArrayLike]:
        """Retrieve the dataset's features (x) and labels (y) for validation as a tuple (x, y)."""
        raise NotImplementedError()

    @abc.abstractmethod
    def test_data(self, **kwargs) -> Tuple[NDArray, ArrayLike]:
        """Retrieve the dataset's features (x) and labels (y) for testing as a tuple (x, y)."""
        raise NotImplementedError()