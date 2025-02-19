import importlib





def print_something():
    config_file = 'configs.pathfinder_512x512_deepglobe' # path to config file for deepglobe
    config = importlib.import_module(config_file)
    model_config = config.model
    print(model_config['type'])





if __name__ == "__main__":
    print_something()


