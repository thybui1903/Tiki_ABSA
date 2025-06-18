# from .registry import Registry

# META_DATASET = Registry("DATASET")

# def build_dataset(config, vocab):
#     data_path = config.path  # config.path phải là string đường dẫn
#     dataset = META_DATASET.get(config.type)(data_path, vocab)
#     return dataset

from .registry import Registry

META_DATASET = Registry("DATASET")

def build_dataset(config, vocab):
    """
    Build dataset from configuration
    
    Args:
        config: Configuration object with 'type' and 'path' attributes
        vocab: Vocabulary object
    
    Returns:
        Dataset instance
    """
    if hasattr(config, 'path'):
        data_path = config.path
    elif isinstance(config, dict) and 'path' in config:
        data_path = config['path']
    else:
        raise ValueError("Configuration must have 'path' attribute or key")
    
    if hasattr(config, 'type'):
        dataset_type = config.type
    elif isinstance(config, dict) and 'type' in config:
        dataset_type = config['type']
    else:
        raise ValueError("Configuration must have 'type' attribute or key")
    
    # Get dataset class from registry and instantiate
    dataset_class = META_DATASET.get(dataset_type)
    dataset = dataset_class(data_path, vocab)
    
    return dataset