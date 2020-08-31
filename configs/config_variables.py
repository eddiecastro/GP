from os.path import curdir, join
from configparser import ConfigParser

Config = ConfigParser()
config_file_path = join(curdir, 'configs', 'dev.ini')
Config.read(config_file_path)

KAFKA_HOST = Config.get("ENV_VARIABLES", "kafka_host")
model_file_name = Config.get("MODEL", "model_file")

PATH = Config.get("FILE_PATHS", "path")
input_file = Config.get("INPUT_DATA", "input_file")
col_x = Config.get("INPUT_DATA", "col_x")
col_y = Config.get("INPUT_DATA", "col_y")
model_file_path = Config.get("MODEL", "model_file")

uri = Config.get("MLFLOW", "uri")
experiment = Config.get("MLFLOW","experiment")

