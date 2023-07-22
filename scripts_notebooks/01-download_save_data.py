from src.generate_dataset_pipeline import generate_dataset
from src.train import *
import time
import torch
import numpy as np
from pathlib import Path

def modify_config(config):
    for key in config.keys():
        if key.endswith("_temp"):
            new_key = "model__" + key[:-5]
            print("Replacing value from key", key, "to", new_key)
            if config[key] == "None":
                config[new_key] = None
            else:
                config[new_key] = config[key]
     
    return config

DATA_PATH = Path("data")

def download_save_data(config=None):
    print("GPU?")
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    #    print(torch.cuda.current_device())
    #    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    print("#####")
    CONFIG_DEFAULT = {"train_prop": 0.70,
                      "val_test_prop": 0.3,
                      "max_val_samples": 50000,
                      "max_test_samples": 50000}

    config.update(CONFIG_DEFAULT)
    print(config)
    # Modify the config in certain cases
    config = modify_config(config)
    
    # print(config)
    DATA_PATH.mkdir(exist_ok=True)
    save_path = DATA_PATH.joinpath(config['dataset_name'])
    save_path.mkdir(exist_ok=True)
    # try:
    if config["n_iter"] == "auto":
        x_train, x_val, x_test, y_train, y_val, y_test, categorical_indicator = generate_dataset(config, np.random.RandomState(0))
        if x_test.shape[0] > 6000:
            n_iter = 1
        elif x_test.shape[0] > 3000:
            n_iter = 2
        elif x_test.shape[0] > 1000:
            n_iter = 3
        else:
            n_iter = 5
    else:
        n_iter = config["n_iter"]
        
    for i in range(n_iter):
        rng = np.random.RandomState(i)
        print(rng.randn(1))
        t = time.time()
        x_train, x_val, x_test, y_train, y_val, y_test, categorical_indicator = generate_dataset(config, rng)
        data_generation_time = time.time() - t
        print("Data generation time:", data_generation_time)
        # print(y_train)
        print(x_train.shape)

        if config["regression"]:
            y_train, y_val, y_test = y_train.astype(np.float32), y_val.astype(np.float32), y_test.astype(
                np.float32)
        else:
            y_train, y_val, y_test = y_train.reshape(-1), y_val.reshape(-1), y_test.reshape(-1)
            # y_train, y_val, y_test = y_train.astype(np.float32), y_val.astype(np.float32), y_test.astype(np.float32)
        x_train, x_val, x_test = x_train.astype(np.float32), x_val.astype(np.float32), x_test.astype(
            np.float32)
        np.save(save_path/f"x_train_fold_{i}.npy", x_train)
        np.save(save_path/f"x_val_fold_{i}.npy", x_val)
        np.save(save_path/f"x_test_fold_{i}.npy", x_test)
        np.save(save_path/f"y_train_fold_{i}.npy", y_train)
        np.save(save_path/f"y_val_fold_{i}.npy", y_val)
        np.save(save_path/f"y_test_fold_{i}.npy", y_test)
        np.save(save_path/f"categorical_indicator_fold_{i}.npy", categorical_indicator)
        np.save(save_path/f"config_fold_{i}.npy", config)



if __name__ == """__main__""":
    # For any dataset from openml, fill in the data__keyword, 
    # dataset_name, data__regression, and data__categorical
    # https://huggingface.co/datasets/inria-soda/tabular-benchmark
    config = {
        'max_test_samples': 50000.0,
        'max_train_samples': 10000.0,
        'max_val_samples': 50000.0,
        'train_prop': 0.7,
        'val_test_prop': 0.3,
        'n_iter': "auto",
        'data__keyword': '1485',  # Update here
        'dataset_name': 'madelon_synthetic', # Update here  
        'data__regression': 0.0, # Update here
        'data__categorical': 0.0, # Update here
        }
    config['regression'] = config['data__regression']
    download_save_data(config)