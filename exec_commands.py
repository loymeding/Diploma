import os
from src.config import Config
from dotenv import load_dotenv


load_dotenv()
project_dir = os.getenv('project_dir')
config = Config()
command_file_path = project_dir + '/commands.yaml'
config.load(command_file_path)
config.run()