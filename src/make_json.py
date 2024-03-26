import argparse
import yaml
import os
from dotenv import load_dotenv


def add_dataloaders_command(config_path: str, dataloaders_path: str) -> list:
    return [{"include": dataloaders_path},
            {"execute": f"{dataloaders_path} get_dataloader {config_path} batch_size {config_path} shuffle {config_path} train_dir"},
            {"move": f"{config_path} train_loader {dataloaders_path} __result__"},
            {"execute": f"{dataloaders_path} get_dataloader {config_path} batch_size {config_path} shuffle {config_path} test_dir"},
            {"move": f"{config_path} test_loader {dataloaders_path} __result__"},
            {"execute": f"{dataloaders_path} get_dataloader {config_path} batch_size {config_path} shuffle {config_path} valid_dir"},
            {"move": f"{config_path} valid_loader {dataloaders_path} __result__"}]


def add_model_command(config_path: str, model_path: str) -> list:
    return [{"include": model_path},
            {"execute": f"{model_path} get_model {config_path} num_classes"},
            {"move": f"{config_path} model {model_path} __result__"}]


def add_train_command(config_path: str, train_path: str) -> list:
    return [{"include": train_path},
            {"execute": f"{train_path} train {config_path} model {config_path} train_loader {config_path} valid_loader \
            {config_path} criterion {config_path} optimizer {config_path} accuracy_fn {config_path} num_epochs {config_path} device"},
            {"move": f"{config_path} metrics {train_path} __result__"}]


def add_predict_command(config_path: str, predict_path: str) -> list:
    return [{"include": predict_path},
            {"execute": f"{predict_path} predict {config_path} model {config_path} test_loader {config_path} device"},
            {"move": f"{config_path} predictions {predict_path} __result__"},
            {"include": "builtins"},
            {"execute": f"builtins print {config_path} predictions"}]


def make_json(
        config_path: str,
        dataloaders_path: str,
        model_path: str,
        train_path: str,
        predict_path: str
):
    data = []
    if config_path:
        data.append({"include": config_path})
    if dataloaders_path:
        data.extend(add_dataloaders_command(config_path, dataloaders_path))
    if model_path:
        data.extend(add_model_command(config_path, model_path))
    if train_path:
        data.extend(add_train_command(config_path, train_path))
    if predict_path:
        data.extend(add_predict_command(config_path, predict_path))

    load_dotenv()
    project_dir = os.getenv('project_dir')
    command_file_path = project_dir + '/commands.yaml'

    with open(command_file_path, 'a') as file:
        yaml.dump(data, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--dataloaders_path", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--train_path", default=None)
    parser.add_argument("--predict_path", default=None)
    args = parser.parse_args()
    make_json(args.config_path, args.dataloaders_path, args.model_path, args.train_path, args.predict_path)
