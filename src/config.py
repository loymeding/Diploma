import sys
import importlib.util

from types import FunctionType
from typing import Any

from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


class Config:
    def __init__(self) -> None:
        self.__config: list[dict[str, str]]

    def load(self, path: str) -> None:
        with open(path, 'r') as file:
            self.__config = load(file, Loader)

    def dump(self, path: str) -> None:
        with open(path, 'w') as file:
            dump(self.__config, file, Dumper)

    def add(self, command: str, *args) -> int:
        self.__config.append({command: ' '.join(args)})
        return len(self.__config)

    def add_include(self, modules: str | list[str]) -> int:
        return self.add('include', *modules if isinstance(modules, list) else modules)

    def add_exclude(self, modules: str | list[str]) -> int:
        return self.add('exclude', *modules if isinstance(modules, list) else modules)

    def add_execute(self, function: FunctionType, args: list[list[str, str]]) -> int:
        return self.add('execute',
                        *[str(function.__module__), str(function.__name__)].extend(list(' '.join(arg) for arg in args)))

    def add_move(self, dst_module: str, dst_object: str, src_module: str, src_object: str) -> int:
        return self.add('move', *(dst_module, dst_object, src_module, src_object))

    def insert(self, line: int, command: str, *args) -> None:
        self.__config.insert(line, {command: ' '.join(args)})

    def insert_include(self, modules: str | list[str]) -> None:
        self.insert('include', *modules if isinstance(modules, list) else modules)

    def insert_exclude(self, modules: str | list[str]) -> None:
        self.insert('exclude', *modules if isinstance(modules, list) else modules)

    def insert_execute(self, function: FunctionType, args: list[list[str, str]]) -> None:
        self.insert('execute',
                    *[str(function.__module__), str(function.__name__)].extend(list(' '.join(arg) for arg in args)))

    def insert_move(self, dst_module: str, dst_object: str, src_module: str, src_object: str) -> None:
        self.insert('move', *(dst_module, dst_object, src_module, src_object))

    def remove(self, line: int) -> dict[str, str]:
        return self.__config.pop(line)

    def run(self) -> None:
        for line in self.__config:
            command, args = tuple(line.items())[0]

            command = command.lower()
            args = args.split()

            if command == 'include':
                for arg in args:
                    self.include(arg)

            if command == 'exclude':
                for arg in args:
                    self.exclude(arg)

            if command == 'execute':
                self.execute(args[0], args[1], args[2:])

            if command == 'move':
                self.set(args[0], args[1], self.get(args[2], args[3]))

    def include(self, name: str) -> None:
        if not name in sys.modules:
            spec = importlib.util.find_spec(name)

            if not spec is None:
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module

                spec.loader.exec_module(module)

    def exclude(self, name: str) -> None:
        if name in sys.modules:
            sys.modules.pop(name)

    def execute(self, module_name: str, object_name: str, args: list[str]) -> None:
        args_value: list[Any] = list()

        for i in range(0, len(args), 2):
            args_value.append(self.get(args[i], args[i + 1]))

        self.set(module_name, '__result__', self.get(module_name, object_name)(*args_value))

    def get(self, module_name: str, object_name: str) -> Any:
        if module_name in sys.modules:
            module = sys.modules[module_name]

            if object_name in dir(module):
                return getattr(module, object_name)

    def set(self, module_name: str, object_name: str, value: Any) -> None:
        if module_name in sys.modules:
            module = sys.modules[module_name]
            setattr(module, object_name, value)
