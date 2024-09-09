import json
import random
from math import sqrt
from typing import Any

from ..tools_support.schema import jsonschema_for_function


def get_weather_at(lon: float, lat: float) -> dict[str, Any]:
    """
    Get the current weather in a given location. All units are in metric system.

    Args:
        lon (float): The longitude of the location.
        lat (float): The latitude of the location.

    Returns:
        Dict[str, Any]: The weather data. Contains the following fields:
            - temperature (float): The temperature in degrees Celsius.
            - humidity (float): The relative humidity.
            - wind_speed (float): The wind speed in meters per second.
            - weather (str): A description of the weather.
    """

    return {
        "temperature": random.uniform(-20, 40),
        "humidity": random.uniform(0, 100),
        "wind_speed": random.uniform(0, 20),
        "weather": random.choice(["sunny", "cloudy", "rainy", "snowy"]),
    }


async def get_location_coordinates(location: str) -> dict[str, float]:
    """
    Get the coordinates of a given location.

    Args:
        location (str): The name of the location.

    Returns:
        Dict[str, float]: The coordinates of the location. Contains the following fields:
            - lon (float): The longitude of the location.
            - lat (float): The latitude of the location.
    """

    return {
        "lon": random.uniform(-180, 180),
        "lat": random.uniform(-90, 90),
    }


def text_complexity(text: str) -> float:
    """Computes complexity score of the given text."""

    return sqrt(len(text))


if __name__ == "__main__":
    for func in [get_weather_at, get_location_coordinates]:
        schema = jsonschema_for_function(func)
        print(json.dumps(schema, indent=2))
