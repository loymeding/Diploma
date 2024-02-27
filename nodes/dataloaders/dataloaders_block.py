from nodes.dataloaders.dataloader_params import DataloadersParams
from nodes.main_node.read_config import read_yaml_schema
from marshmallow_dataclass import class_schema
import sys


class DataloadersBlock():
    def __init__(self, config_path):
        DataloadersParamsSchema = class_schema(DataloadersParams)
        self.params = read_yaml_schema(config_path, DataloadersParamsSchema)
        module_path = self.params.module_name
        func_name = self.params.func_name
        sys.path.append(module_path)
        module = __import__(self.params.module_name)
        get_dataloaders = getattr(module, func_name)
        self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(self.params)