import numpy as np
from typing import Any, Dict, List, Union
import json

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert NumPy types to Python native types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

def safe_json_dumps(obj: Any) -> str:
    """
    Safely serialize object to JSON string, handling NumPy types.
    """
    return json.dumps(obj, cls=NumpyEncoder)

def prepare_for_json(data: Union[Dict, List, Any]) -> Union[Dict, List, Any]:
    """
    Prepare data for JSON serialization by converting all NumPy types.
    """
    return convert_numpy_types(data)