import gc
import uuid
from pathlib import Path
import os
import joblib
import numpy as np
import openml
import pandas as pd
import requests
import torch
import torch.nn as nn
from captum.attr import (
    DeepLift,
    GradientShap,
)
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import GANDALFConfig
from sklearn.metrics import accuracy_score, r2_score

from non_informative_features_experiment import (
    get_best_gflu_params,
    get_best_xgb_params,
    load_data,
    train_GFLU_model,
    train_xgb_model,
)

DATA_DIR = Path("data")
OUTPUT_PATH = Path("output")
TRAINING_DONE_TOKEN = os.getenv("TELEGRAM_TOKEN", default=None)
TRAINING_UPDATE_TOKEN = os.getenv("TELEGRAM_TOKEN", default=None)
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", default=None)
NOTIFY_TELEGRAM = TRAINING_DONE_TOKEN is not None and TELEGRAM_CHAT_ID is not None


def notify_telegram(message, done=False):
    try:
        requests.get(
            f"https://api.telegram.org/bot{TRAINING_DONE_TOKEN if done else TRAINING_UPDATE_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={message}&parse_mode=html"
        )
    except Exception as e:
        raise e


class InterpModel(nn.Module):
    def __init__(self, model, num_cols):
        super().__init__()
        self.model = model
        self.num_cols = num_cols

    def forward(self, input):
        in_dict = dict(
            continuous=input[:, : self.num_cols],
            target=None,
            categorical=input[:, self.num_cols :],
        )
        return self.model(in_dict)["logits"]


def _permutation_importance(model, val, rng):
    if model.config["task"] == "regression":

        def metric(val, model):
            return r2_score(
                val["target"],
                model.predict(val, verbose=False)[
                    f"{model.config['target'][0]}_prediction"
                ],
            )

    else:

        def metric(val, model):
            return accuracy_score(
                val["target"], model.predict(val, verbose=False).prediction
            )

    baseline = metric(val, model)
    imp = []
    for col in val.columns:
        if col == "target":
            continue
        save = val[col].copy()
        val[col] = rng.permutation(val[col])
        m = metric(val, model)
        val[col] = save
        imp.append(baseline - m)
    return np.array(imp)


def permutation_importance(model, val, n_repeats, seed=42):
    imps = []
    rng = np.random.default_rng(seed)
    for _ in range(n_repeats):
        imps.append(_permutation_importance(model, val, rng))
    imps = np.array(imps)
    imp_means = imps.mean(axis=0)
    imp_stds = imps.std(axis=0)
    return imp_means, imp_stds


def enrich_config(config):
    data_path = DATA_DIR / config["dataset"]
    d_config_files = list(data_path.glob("*config*"))
    d_config = np.load(d_config_files[0], allow_pickle=True).item()
    task = "classification" if d_config["regression"] == 0 else "regression"
    n_folds = len(d_config_files)
    config["n_folds"] = n_folds
    config["task"] = task
    dataset = openml.datasets.get_dataset(
        d_config["data__keyword"], download_data=False
    )
    features = [
        dataset.features[i].name
        for i in range(len(dataset.features))
        if dataset.features[i].name != dataset.default_target_attribute
    ]
    config["target"] = dataset.default_target_attribute
    # config.update(d_config)
    assert (
        d_config.get("data__categorical", 0) != 1
    ), "Categorical features not supported"
    # n_features = np.load(data_path / "x_test_fold_0.npy", allow_pickle=True).shape[1]
    # cat_col_names = None
    num_col_names = features
    # config["cat_col_names"] = cat_col_names
    config["num_col_names"] = num_col_names
    metric_name = "test_r2_score" if task == "regression" else "test_accuracy"
    config["metric_name"] = metric_name
    return config


def main(config):
    config = enrich_config(config)
    xgb_params = get_best_xgb_params(config)
    gflu_params = get_best_gflu_params(config)
    # using first fold
    train, val, test = load_data(config, fold=0)
    train.columns = config["num_col_names"] + ["target"]
    val.columns = config["num_col_names"] + ["target"]
    test.columns = config["num_col_names"] + ["target"]
    xgb_model, val_score, test_score = train_xgb_model(
        config, config["num_col_names"], xgb_params, train, val, test
    )
    gflu_model, val_score, test_score = train_GFLU_model(
        config, config["num_col_names"], gflu_params, train, val, test
    )
    gflu_model.save_model(f"saved_models/tuned_gflu_{config['dataset']}")
    gflu_imp_1 = gflu_model.feature_importance().importance.values
    
    
    test_dl = gflu_model.datamodule.prepare_inference_dataloader(test.sample(5000))
    tensor_inp = []
    for x in test_dl:
        tensor_inp.append(torch.cat((x["continuous"], x["categorical"]), dim=1))

    tensor_inp = torch.cat(tensor_inp, dim=0)

    tensor_inp_tr = []
    for x in gflu_model.datamodule.train_dataloader():
        tensor_inp_tr.append(torch.cat((x["continuous"], x["categorical"]), dim=1))

    tensor_inp_tr = torch.cat(tensor_inp_tr, dim=0)
    del train, test
    gc.collect()
    is_reg = config["task"] == "regression"
    mdl = InterpModel(gflu_model.model, len(config["num_col_names"]))
    gs = GradientShap(mdl)
    dl = DeepLift(mdl)
    gs_attr_test = (
        gs.attribute(tensor_inp, tensor_inp_tr, target=None if is_reg else 1)
        .detach()
        .numpy()
    )
    dl_attr_test = (
        dl.attribute(tensor_inp, target=None if is_reg else 1).detach().numpy()
    )
    perm_imp, perm_imp_std = permutation_importance(gflu_model, val, n_repeats=5)
    # get igx, dls, etc importances into a single dataframe
    d = {
        "Features": config["num_col_names"],
        "GradientShap": np.mean(gs_attr_test, axis=0),
        "DeepLift": np.mean(dl_attr_test, axis=0),
        "PermutationImportance": perm_imp,
        "PermutationImportance_Std": perm_imp_std,
        "GFLU_Imp_1": gflu_imp_1,
        "XGBoost": xgb_model.feature_importances_,
    }
    interp_df = pd.DataFrame(d)
    interp_df["Dataset"] = config["dataset"]
    return interp_df


if __name__ == "__main__":
    config = {
        "dataset": "yprop_4_1",
        "batch_size": 512,
        "max_epochs": 100,
        "early_stopping_patience": 10,
        "optimizer": "AdamW",
    }
    data_summary_df = pd.read_parquet("data/data_summary.parquet")
    ds = data_summary_df.loc[
        (data_summary_df.n_features.between(20, 60))
        & (~data_summary_df.has_categorical_features)
        & (data_summary_df.dataset_name != "pol_regression"),
        "dataset_name",
    ].tolist()
    print(ds)
    for d in ds:
        config["dataset"] = d
        main(config).to_parquet(OUTPUT_PATH / f"feat_imp_{config['dataset']}.parquet")
        if NOTIFY_TELEGRAM:
            notify_telegram(f"Interpretability Run for {config['dataset']} is finished")
    if NOTIFY_TELEGRAM:
        notify_telegram(f"Interpretability Run is finished", done=True)
