from dataclasses import dataclass, field


@dataclass()
class ModelParams:
    module_name: str
    func_name: str
    pretrained: bool = field(default=True)
