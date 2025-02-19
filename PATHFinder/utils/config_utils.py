import importlib



def load_config(config_path: str):
    """
        Loads config file from config path

        Returns: 
                config file
    """
    config = importlib.import_module(config_path)
    return config





def build_model(config_path:str)

    """
        Builds model from config file
    """

    config = load_config(config_path)






