import time
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import wandb
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import GANDALFConfig

os.environ["WANDB_MODE"] = "offline"
DATA_DIR = Path("data")
EXP_RES = "data/tuning_results_best_rows.parquet"
TRAINING_DONE_TOKEN = os.getenv("TELEGRAM_TOKEN", default=None)
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", default=None)
NOTIFY_TELEGRAM = TRAINING_DONE_TOKEN is not None and TELEGRAM_CHAT_ID is not None
WANDB_PROJECT = "GANDALF"


def notify_telegram(message):
    try:
        requests.get(
            f"https://api.telegram.org/bot{TRAINING_DONE_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={message}&parse_mode=html"
        )
    except Exception as e:
        raise e


def train_model_on_config(config=None):
    print("GPU?")
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    #    print(torch.cuda.current_device())
    #    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    print("#####")

    # "model__use_checkpoints": True} #TODO
    data_path = DATA_DIR / config["dataset"]
    exp_res = pd.read_parquet(EXP_RES)
    d_config_files = list(data_path.glob("*config*"))
    d_config = np.load(d_config_files[0], allow_pickle=True).item()
    task = "classification" if d_config["regression"] == 0 else "regression"
    n_folds = len(d_config_files)
    config["n_folds"] = n_folds
    if d_config.get("data__categorical", 0) == 1:
        categorical_indicator = np.load(
            data_path / "categorical_indicator_fold_0.npy", allow_pickle=True
        )
        feat_names = [f"feature_{i}" for i in range(categorical_indicator.shape[0])]
        cat_col_names = [
            feat_names[i]
            for i in range(len(feat_names))
            if categorical_indicator[i] == 1
        ]
        num_col_names = [
            feat_names[i]
            for i in range(len(feat_names))
            if categorical_indicator[i] == 0
        ]
    else:
        n_features = np.load(
            data_path / "x_test_fold_0.npy", allow_pickle=True
        ).shape[1]
        cat_col_names = None
        num_col_names = [f"feature_{i}" for i in range(n_features)]
    best_params = (
        exp_res.xs(config["dataset"], level="data__keyword")
        .xs("GFLU", level="model_name")["params"]
        .item()
    )
    model_params = {
        k.replace("params_", ""): v
        for k, v in best_params.items()
        if ("optimizer" not in k)
    }
    model_params["gflu_stages"] = int(model_params["gflu_stages"])
    model_params["head_config"] = {"layers": "32-16"}
    weight_decay = best_params["params_optimizer_config__weight_decay"]
    print(f"Using Data from: {data_path} for task: {task} with {n_folds} folds")
    if NOTIFY_TELEGRAM:
        notify_telegram(
            f"Start Training for data: <b>{config['dataset']}</b> for task: {task} with <b>{n_folds}</b> folds."
        )
    data_config = DataConfig(
        target=["target"],
        continuous_cols=num_col_names,
        categorical_cols=cat_col_names or [],
        normalize_continuous_features=True,
    )
    trainer_config = TrainerConfig(
        auto_lr_find=False,  # Runs the LRFinder to automatically derive a learning rate
        batch_size=config["batch_size"],
        max_epochs=config["max_epochs"],
        early_stopping="valid_loss",
        early_stopping_mode="min",  # Set the mode as min because for val_loss, lower is better
        early_stopping_patience=config[
            "early_stopping_patience"
        ],  # No. of epochs of degradation training will wait before terminating
        # early_stopping_min_delta=0.0001,  # Minimum delta for improvement in val_loss
        checkpoints="valid_loss",
        load_best=True,  # After training, load the best checkpoint
        # progress_bar="none",  # Turning off Progress bar
        # trainer_kwargs=dict(enable_model_summary=False),  # Turning off model summary
    )
    optimizer_config = OptimizerConfig(
        optimizer=config["optimizer"], optimizer_params={"weight_decay": weight_decay}
    )
    model_config = GANDALFConfig(
        task=task,
        metrics=["r2_score"] if task == "regression" else None,
        metrics_prob_input=[False] if task == "regression" else None,
        **model_params,
    )
    model_id = uuid.uuid4().hex
    name = f"{config['dataset']}_{model_id}"
    metric_name = "test_r2_score" if task == "regression" else "test_accuracy"
    # Initialize a new wandb run
    with wandb.init(
        config=config,
        mode="online" if config["wandb"] else "disabled",
        project=WANDB_PROJECT,
        name=name,
    ) as run:
        config = wandb.config
        print(config)
        test_scores = []
        times = []
        for fold in range(n_folds):
            start_time = time.time()
            x_train = np.load(
                data_path / f"x_train_fold_{fold}.npy", allow_pickle=True
            )
            y_train = np.load(
                data_path / f"y_train_fold_{fold}.npy", allow_pickle=True
            ).reshape(-1, 1)
            x_val = np.load(data_path / f"x_val_fold_{fold}.npy", allow_pickle=True)
            y_val = np.load(
                data_path / f"y_val_fold_{fold}.npy", allow_pickle=True
            ).reshape(-1, 1)
            x_test = np.load(
                data_path / f"x_train_fold_{fold}.npy", allow_pickle=True
            )
            y_test = np.load(
                data_path / f"y_train_fold_{fold}.npy", allow_pickle=True
            ).reshape(-1, 1)
            # combine x and y into a dataframe
            train = pd.DataFrame(
                np.concatenate([x_train, y_train], axis=1),
                columns=[f"feature_{i}" for i in range(x_train.shape[1])]
                + ["target"],
            )
            val = pd.DataFrame(
                np.concatenate([x_val, y_val], axis=1),
                columns=[f"feature_{i}" for i in range(x_val.shape[1])] + ["target"],
            )
            test = pd.DataFrame(
                np.concatenate([x_test, y_test], axis=1),
                columns=[f"feature_{i}" for i in range(x_test.shape[1])] + ["target"],
            )
            # Initialize the tabular model
            tabular_model = TabularModel(
                data_config=data_config,
                model_config=model_config,
                optimizer_config=optimizer_config,
                trainer_config=trainer_config,
            )
            try:
                # Fit the model
                if name.startswith("Ailerons"):
                    # Need special transformation to target to avoid gradient underflow
                    tabular_model.fit(
                        train=train,
                        validation=val,
                        target_transform=(lambda x: x * 1000, lambda x: x / 1000),
                    )
                else:
                    tabular_model.fit(
                        train=train,
                        validation=val,
                    )
                end_time = time.time()
                result = tabular_model.evaluate(test, verbose=False)
                test_scores.append(result[0][metric_name])
                times.append(end_time - start_time)
            except RuntimeError as e:
                # print(e)
                # gc.collect()
                # torch.cuda.empty_cache()
                if NOTIFY_TELEGRAM:
                    notify_telegram(f"<b>{config['dataset']}</b> run crashed!!.")
                raise e
            if NOTIFY_TELEGRAM:
                notify_telegram(
                    f"<b>{config['dataset']}</b> fold {fold} finished with {result[0][metric_name]} in {np.mean(end_time - start_time)} seconds."
                )
        if n_folds > 1:
            wandb.log(
                {
                    "test_scores": test_scores,
                    "mean_test_score": np.mean(test_scores),
                    "std_test_score": np.std(test_scores),
                    "max_test_score": np.max(test_scores),
                    "min_test_score": np.min(test_scores),
                    "mean_time": np.mean(times),
                    "std_time": np.std(times),
                    "times": times,
                },
                commit=False,
            )
        else:
            wandb.log(
                {
                    "mean_test_score": np.mean(test_scores),
                    "mean_time": end_time - start_time,
                },
                commit=False,
            )
        if NOTIFY_TELEGRAM:
            notify_telegram(
                f"<b>{config['dataset']}</b> run finished with {np.mean(test_scores)} in {np.mean(times)} seconds."
            )


if __name__ == "__main__":
    config = {
        "dataset": "heloc",
        "batch_size": 512,
        "max_epochs": 100,
        "early_stopping_patience": 10,
        "optimizer": "AdamW",
        "wandb": True,
    }
    datasets = [
        "default-of-credit-card-clients",
        "heloc",
        "eye_movements",
        "Higgs",
        "pol",
        "albert",
        "road-safety",
        "MiniBooNE",
        "covertype",
        "jannis",
        "Bioresponse",
        "cpu_act",
        "Ailerons",
        "yprop_4_1",
        "superconduct",
        "Allstate_Claims_Severity",
        "topo_2_1",
        "Mercedes_Benz_Greener_Manufacturing"
        ]
    for col in datasets:
        config["dataset"] = col

        train_model_on_config(config)
