import bz2
import gzip
import lzma
import os
import pickle
from typing import Any, Optional

import pandas as pd

def save_object(obj: Any, path: str, compression: Optional[str] = None) -> None:
    """
    Serialize and save a Python object (e.g., model, transformer, dictionary) to disk,
    with optional compression.

    This function ensures that the destination directory exists before writing,
    supports multiple compression formats, and appends an appropriate file extension
    based on the compression type.

    Supported compression formats:
        - None: Uncompressed `.pickle`
        - 'gzip': GZIP-compressed `.pickle.gz`
        - 'bz2': BZ2-compressed `.pickle.bz2`
        - 'lzma': LZMA/XZ-compressed `.pickle.xz`

    Args:
        obj (Any):
            The Python object to serialize and save.
        path (str):
            Destination file path (without extension).
            Example: `"models/random_forest_model"`
        compression (str, optional):
            Compression type to use ('gzip', 'bz2', 'lzma', or None).
            Defaults to None (uncompressed).
    """
    # Ensure the output directory exists
    root = os.path.dirname(path)
    if root and not os.path.exists(root):
        os.makedirs(root)

    # Handle supported compression formats
    if compression in ["gzip", "bz2", "lzma"]:
        if compression == "gzip":
            ext = ".pickle.gz"
            with gzip.open(path + ext, "wb") as f:
                pickle.dump(obj, f)
        elif compression == "bz2":
            ext = ".pickle.bz2"
            with bz2.BZ2File(path + ext, "wb") as f:
                pickle.dump(obj, f)
        elif compression == "lzma":
            ext = ".pickle.xz"
            with lzma.open(path + ext, "wb") as f:
                pickle.dump(obj, f)

    else:
        # Save as an uncompressed pickle file
        if compression is not None:
            print("Warning: Unknown compression type. Defaulting to uncompressed pickle format.")
        ext = ".pickle"
        with open(path + ext, "wb") as f:
            pickle.dump(obj, f)

    # Print confirmation
    print(f"Successfully saved object to {path + ext}")

    return

def load_object(path: str) -> Any:
    """
    Load and deserialize a Python object (e.g., model, transformer, dictionary)
    from disk, automatically handling compressed pickle formats.

    This function supports the same compression extensions as `save_object()`:
        - `.pickle`: Uncompressed
        - `.pickle.gz`: GZIP-compressed
        - `.pickle.bz2`: BZ2-compressed
        - `.pickle.xz`: LZMA/XZ-compressed

    Args:
        path (str):
            Full path to the serialized object file, including its extension.
            Example: `"models/random_forest_model.pickle.gz"`

    Returns:
        Any:
            The deserialized Python object.
    """
    # Determine compression type based on file extension
    if path.endswith(".pickle.gz"):
        compression = "gzip"
    elif path.endswith(".pickle.bz2"):
        compression = "bz2"
    elif path.endswith(".pickle.xz"):
        compression = "lzma"
    else:
        compression = None

    # Load object using appropriate method
    if compression == "gzip":
        with gzip.open(path, "rb") as f:
            obj = pickle.load(f)
    elif compression == "bz2":
        with bz2.BZ2File(path, "rb") as f:
            obj = pickle.load(f)
    elif compression == "lzma":
        with lzma.open(path, "rb") as f:
            obj = pickle.load(f)
    else:
        with open(path, "rb") as f:
            obj = pickle.load(f)

    # Return the loaded Python object
    print(f"Successfully loaded object from {path}")
    
    return obj

def load_metric_data(path: str, model_type: str) -> pd.DataFrame:
    """
    Load a metrics CSV file, reshape it into a tidy format, and extract
    model hyperparameters from the model name column based on model type.

    Args:
        path (str):
            Filepath to the metrics CSV produced by `fit_models`.
        model_type (str):
            Must be either 'gmm' or 'iforest', determining which parameters
            to extract from the model identifier string.

    Returns:
        pd.DataFrame:
            A tidy dataframe with extracted hyperparameters and metric values.

    Raises:
        ValueError:
            If `model_type` is not one of {'gmm', 'iforest'}.
    """

    # Validate model type
    if model_type not in {'gmm', 'iforest'}:
        raise ValueError(
            f"Invalid model_type '{model_type}'. Expected one of: 'gmm', 'iforest'."
        )

    # Load CSV, set "score" as index, then transpose so each row is a model
    metrics_frame = pd.read_csv(path).set_index('score')
    metrics_frame.index.name = ''
    metrics_frame = metrics_frame.T.reset_index(names = 'model')

    # Extract parameters depending on the model type
    if model_type == 'gmm':
        metrics_frame.insert(
            0,
            'covariance_type',
            metrics_frame.model.str.extract(r'covariance_type_([a-z]+)-')
        )
        metrics_frame.insert(
            1,
            'n_components',
            metrics_frame.model.str.extract(r'n_components_(\d+)')
        )
        metrics_frame = metrics_frame.drop(columns = 'model')

    elif model_type == 'iforest':
        metrics_frame.insert(
            0,
            'contamination_rate',
            metrics_frame.model.str.extract(r'contamination_([\d\.]+)-')
        )
        metrics_frame.insert(
            1,
            'max_features',
            metrics_frame.model.str.extract(r'max_features_([\d\.]+)')
        )
        metrics_frame.insert(
            2,
            'max_samples',
            metrics_frame.model.str.extract(r'max_samples_([\d]+)')
        ) 
        metrics_frame = metrics_frame.drop(columns = 'model')

    return metrics_frame