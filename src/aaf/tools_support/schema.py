import inspect
import json
from typing import Any, Callable, Dict, Optional, get_args, get_origin

import typing_inspect
from docstring_parser import parse


def jsonschema_for_function(func: Callable, parameters_key: str = "parameters") -> Dict[str, Any]:
    """
    Generate a JSON schema for the given function.

    Args:
        func (Callable): The function to generate a schema for.
        parameters_key (str, optional): The key to use for the parameters. Defaults to "parameters".

    Returns:
        Dict[str, Any]: A dictionary representing the JSON schema.
    """
    schema = {
        "name": func.__name__,
        "description": "",
        parameters_key: {"type": "object", "properties": {}, "required": []},
    }

    # Parse the docstring
    docstring = parse(inspect.getdoc(func) or "")
    schema["description"] = docstring.short_description or ""

    # Get function signature
    sig = inspect.signature(func)

    for name, param in sig.parameters.items():
        param_schema = {}

        # Get type information
        if param.annotation != inspect.Parameter.empty:
            if typing_inspect.is_optional_type(param.annotation):
                type_name = _get_type_name(typing_inspect.get_args(param.annotation)[0])
            else:
                type_name = _get_type_name(param.annotation)

            if type_name.startswith("array["):
                item_type = type_name[6:-1]  # Extract inner type
                param_schema["type"] = "array"
                param_schema["items"] = {"type": item_type}
            else:
                param_schema["type"] = type_name

        # Get description from docstring
        param_doc = next((p for p in docstring.params if p.arg_name == name), None)
        if param_doc:
            param_schema["description"] = param_doc.description

        # Check if parameter is required
        if param.default == inspect.Parameter.empty:
            schema[parameters_key]["required"].append(name)

        schema[parameters_key]["properties"][name] = param_schema

    return schema


def _get_type_name(typ: Any) -> str:
    """Helper function to get a string representation of a type."""
    if get_origin(typ) == list:
        item_type = get_args(typ)[0]
        return f"array[{_get_type_name(item_type)}]"
    if typ == float:
        return "number"
    if typ == str:
        return "string"
    if typ == int:
        return "integer"
    if typ == bool:
        return "boolean"
    if typ == dict:
        return "object"
    if hasattr(typ, "__name__"):
        return typ.__name__.lower()
    return str(typ).lower()


if __name__ == "__main__":

    # Example usage:
    def get_current_weather(location: str, unit: Optional[str] = "celsius") -> Dict[str, Any]:
        """
        Get the current weather in a given location.

        Args:
            location (str): The city and state, e.g. San Francisco, CA
            unit (str, optional): The unit of temperature. Defaults to "celsius".

        Returns:
            Dict[str, Any]: A dictionary containing weather information.
        """
        pass

    schema = jsonschema_for_function(get_current_weather)
    print(json.dumps(schema, indent=2))
