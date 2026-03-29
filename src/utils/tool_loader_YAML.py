# This module defines a function to load tools from a YAML configuration file.
# It uses the 'yaml' library to read the configuration and 'importlib' to dynamically import the specified modules and functions.

import importlib   # allows loading Python modules/functions dynamically from strings
import yaml        # reads YAML configuration files (e.g., tools.yaml)

def load_tools(config_path="configs/tools.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    registry = {}

    for tool_name, spec in config["tools"].items():
        module = importlib.import_module(spec["module"])
        func = getattr(module, spec["function"])
        registry[tool_name] = {
            "func": func,
            "description": spec.get("description", ""),
            "parameters": spec.get("parameters", {})
        }

    return registry