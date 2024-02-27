import yaml
from typing import Optional
from marshmallow_dataclass import class_schema


def read_yaml_dict(path: str) -> dict:
    with open(path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict


def read_yaml_schema(path: str, schema: Optional[class_schema]) -> Optional[class_schema]:
    with open(path, 'r') as file:
        config_dict = yaml.safe_load(file)
        config_schema = schema().load(config_dict)
    return config_schema
