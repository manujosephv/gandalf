import gc
import os
from copy import deepcopy

import numpy as np
import optuna
import pandas as pd
import requests
import torch

# from optuna.integration import PyTorchLightningPruningCallback
from pytorch_tabular import TabularModel

TRAINING_UPDATE_TOKEN = os.getenv("TELEGRAM_TOKEN", None)
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", None)


class OptunaTuner:
    def __init__(
        self,
        trainer_config,
        optimizer_config,
        model_config,
        data_config,
        data_path,
        direction,
        metric_name,
        search_space_fn,
        pruning=True,
        strategy="random",
        notify_progress=True,
        **kwargs,
    ):
        self.trainer_config = trainer_config
        self.optimizer_config = optimizer_config
        self.model_config = model_config
        self.data_config = data_config
        self.data_path = data_path
        self.search_space_fn = search_space_fn
        self.direction = direction
        self.metric_name = metric_name
        self.pruning = pruning
        if TRAINING_UPDATE_TOKEN is None or TELEGRAM_CHAT_ID is None:
            notify_progress = False
        self.notify_progress = notify_progress
        pruner = (
            optuna.pruners.PercentilePruner(
                percentile=75.0, n_startup_trials=10, n_warmup_steps=5
            )
            if pruning
            else optuna.pruners.NopPruner()
        )
        # self._uuid = uuid.uuid4().hex

        kwargs["direction"] = direction
        kwargs["pruner"] = kwargs.get("pruner", pruner)
        kwargs["load_if_exists"] = True
        kwargs["sampler"] = (
            optuna.samplers.TPESampler(seed=42)
            if strategy == "tpe"
            else optuna.samplers.RandomSampler(seed=42)
        )
        self.optuna_study_kwargs = kwargs
        self.study = optuna.create_study(**self.optuna_study_kwargs)
        self.n_fold = len(list(data_path.glob("*config*")))

    def _notify_progress(self, trial, value):
        try:
            message = f"Trial {trial.number} finished with value: <b>{value:.5f}</b> and params: {trial.params}.\n\n"
            if trial.number > 0:
                message += f"<b>Best Trial</b>: {self.study.best_trial.number} with value: <b>{self.study.best_trial.value:.5f}</b> and params: {self.study.best_trial.params}"
            requests.get(
                f"https://api.telegram.org/bot{TRAINING_UPDATE_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={message}&parse_mode=html"
            )
        except Exception:
            pass

    def objective(self, trial):
        trainer_config_t = deepcopy(self.trainer_config)
        optimizer_config_t = deepcopy(self.optimizer_config)
        model_config_t = deepcopy(self.model_config)

        params = self.search_space_fn(trial)
        print(f"Trial Params: {params}")
        # Now set the parameters to the right config
        for name, param in params.items():
            if name.startswith("optimizer"):
                optimizer_config_t.optimizer_params["weight_decay"] = param
            else:
                setattr(model_config_t, name, param)

        metrics = []
        for fold in range(self.n_fold):
            x_train = np.load(
                self.data_path / f"x_train_fold_{fold}.npy", allow_pickle=True
            )
            y_train = np.load(
                self.data_path / f"y_train_fold_{fold}.npy", allow_pickle=True
            ).reshape(-1, 1)
            x_val = np.load(
                self.data_path / f"x_val_fold_{fold}.npy", allow_pickle=True
            )
            y_val = np.load(
                self.data_path / f"y_val_fold_{fold}.npy", allow_pickle=True
            ).reshape(-1, 1)
            x_test = np.load(
                self.data_path / f"x_train_fold_{fold}.npy", allow_pickle=True
            )
            y_test = np.load(
                self.data_path / f"y_train_fold_{fold}.npy", allow_pickle=True
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
                data_config=self.data_config,
                model_config=model_config_t,
                optimizer_config=optimizer_config_t,
                trainer_config=trainer_config_t,
            )
            try:
                # Fit the model
                if self.study.study_name.startswith("Ailerons"):
                    # Need special transformation to target to avoid gradient underflow
                    tabular_model.fit(
                        train=train,
                        validation=val,
                        target_transform=(lambda x: x*1000, lambda x: x/1000)
                        # callbacks=[
                        #     PyTorchLightningPruningCallback(trial, monitor="valid_loss")
                        # ],
                    )
                else:
                    tabular_model.fit(
                        train=train,
                        validation=val,
                        # callbacks=[
                        #     PyTorchLightningPruningCallback(trial, monitor="valid_loss")
                        # ],
                    )
                result = tabular_model.evaluate(test, verbose=False)
                metrics.append(result[0][self.metric_name])
            except RuntimeError as e:
                print(e)
                gc.collect()
                torch.cuda.empty_cache()
                metrics.append(0 if self.direction == "maximize" else 1e6)
        ret_metric = np.nanmean(metrics)
        print("Trial Value: ", ret_metric)
        if self.notify_progress:
            self._notify_progress(trial, ret_metric)
        return ret_metric

    def tune(self, n_trials, timeout=None, n_jobs=1):
        self.study.optimize(
            self.objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs
        )
        # Return the best params, trials dataframe and best value
        return (
            self.study.best_params,
            self.study.best_value,
            self.study.trials_dataframe(),
        )
