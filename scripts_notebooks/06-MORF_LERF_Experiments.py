import gc
import uuid
from pathlib import Path

import joblib
import numpy as np
import openml
import pandas as pd
import requests
import torch
import torch.nn as nn
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import GANDALFConfig
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier

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

def _get_metric(model):
    if isinstance(model, TabularModel):
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
    else: #XGBoost
        if isinstance(model, XGBClassifier):
            def metric(val, model):
                return accuracy_score(
                    val["target"], model.predict(val[model.feature_names_in_])
                )
        else:
            def metric(val, model):
                return r2_score(
                    val["target"], model.predict(val[model.feature_names_in_])
                )
    return metric



def _perturb_score(model, val, cols, rng):
    metric = _get_metric(model)

    baseline = metric(val, model)
    scores = [baseline]
    for col in cols:
        val[col] = rng.permutation(val[col])
        scores.append(metric(val, model))
    return np.array(scores)


def perturb_imp_features(model, val, n_repeats, cols, seed=42):
    imps = []
    rng = np.random.default_rng(seed)
    for _ in range(n_repeats):
        imps.append(_perturb_score(model, val.copy(deep=True), cols, rng))
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
    if Path(f"saved_models/tuned_gflu_{config['dataset']}").exists():
        gflu_model = TabularModel.load_model(f"saved_models/tuned_gflu_{config['dataset']}")
    else:
        gflu_model, val_score, test_score = train_GFLU_model(
            config, config["num_col_names"], gflu_params, train, val, test
        )
        gflu_model.save_model(f"saved_models/tuned_gflu_{config['dataset']}")
    interp_df = pd.read_parquet(OUTPUT_PATH/f"feat_imp_{config['dataset']}.parquet").set_index("Features")
    topn = min(config['morf_topn'], len(interp_df))
    morf_order = interp_df["GFLU_Imp_1"].sort_values(ascending=False).head(topn).index.tolist()
    morf_scores, morf_scores_std = perturb_imp_features(gflu_model, val, n_repeats=5, cols=morf_order)
    # morf_scores = perturb_imp_features(gflu_model, val, n_repeats=5, cols=morf_order)
    morf_order = interp_df["XGBoost"].sort_values(ascending=False).head(topn).index.tolist()
    xgb_morf_scores, xgb_morf_scores_std = perturb_imp_features(xgb_model, val, n_repeats=5, cols=morf_order)
    # xgb_morf_scores = perturb_imp_features(xgb_model, val, n_repeats=5, cols=morf_order)
    lorf_order = interp_df["GFLU_Imp_1"].sort_values(ascending=True).head(topn).index.tolist()
    lorf_scores, lorf_scores_std = perturb_imp_features(gflu_model, val, n_repeats=5, cols=lorf_order)
    # lorf_scores = perturb_imp_features(gflu_model, val, n_repeats=5, cols=lorf_order)
    lorf_order = interp_df["XGBoost"].sort_values(ascending=True).head(topn).index.tolist()
    xgb_lorf_scores, xgb_lorf_scores_std = perturb_imp_features(xgb_model, val, n_repeats=5, cols=lorf_order)
    # xgb_lorf_scores = perturb_imp_features(xgb_model, val, n_repeats=5, cols=lorf_order)
    d = {
        "N": np.arange(morf_scores.shape[0]),
        "morf": morf_scores,
        "morf_std": morf_scores_std,
        "lorf": lorf_scores,
        "lorf_std": lorf_scores_std,
        "xgb_morf": xgb_morf_scores,
        "xgb_morf_std": xgb_morf_scores_std,
        "xgb_lorf": xgb_lorf_scores,
        "xgb_lorf_std": xgb_lorf_scores_std,
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
        "morf_topn": 15,
        "reps": 5,
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
        main(config).to_parquet(OUTPUT_PATH / f"morf_lorf_{config['dataset']}.parquet")
        if NOTIFY_TELEGRAM:
            notify_telegram(f"MORF/LORF Run for {config['dataset']} is finished")
    if NOTIFY_TELEGRAM:
        notify_telegram(f"MORF/LORF Run is finished", done=True)
