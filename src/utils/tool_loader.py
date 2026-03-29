"""
Load tools directly from Python modules.

Functions decorated with @tool are automatically detected.
"""

import importlib
import inspect


def load_tools(module_paths):
    """
    Load LangChain/LangGraph tools from modules.

    Parameters
    ----------
    module_paths : list[str]
        List of module paths containing tool functions.

    Returns
    -------
    dict
        Tool registry.
    """

    registry = {}

    for module_path in module_paths:

        module = importlib.import_module(module_path)

        for _, obj in inspect.getmembers(module):

            # LangChain @tool returns a Tool object
            if hasattr(obj, "name") and hasattr(obj, "description"):

                registry[obj.name] = {
                    "func": obj,
                    "description": obj.description,
                    "parameters": getattr(obj, "args", {})
                }

    return registry