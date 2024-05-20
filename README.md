# GANDALF: Gated Adaptive Network for Deep Automated Learning of Features

## Download Datasets

Run scripts/01-download_save_data.py with the right configs to download the datasets.

## Download Tabular Benchmark
```curl -L -o analyses/results/benchmark_total.csv https://figshare.com/ndownloader/files/40081681```

## Run Experiments

Run the scripts and notebooks in the `scripts_notebooks` folder in the order mentioned to reproduce the experiments.

## Model Implementation

The model is implemented and added to PyTorch Tabular as one of the many models implemented there.
`https://github.com/manujosephv/pytorch_tabular/tree/main/src/pytorch_tabular/models/gandalf`

The code is also available in the `src` folder of this repo, but more to show the implementation. The actual implementation is in the PyTorch Tabular repo.

### Usage:

```python
from pytorch_tabular import TabularModel
from pytorch_tabular.models import GANDALFConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)

data_config = DataConfig(
    target=[
        "target"
    ],  # target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
    continuous_cols=num_col_names,
    categorical_cols=cat_col_names,
)
trainer_config = TrainerConfig(
    auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
    batch_size=1024,
    max_epochs=100,
)
optimizer_config = OptimizerConfig()

model_config = GANDALFConfig(
    task="classification",
    gflu_stages=15,
    gflu_dropout=0.01,
    gflu_feature_init_sparsity=0.3,
    learnable_sparsity=True,
    learning_rate=1e-3,
)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)
tabular_model.fit(train=train, validation=val)
result = tabular_model.evaluate(test)
pred_df = tabular_model.predict(test)
tabular_model.save_model("examples/basic")
loaded_model = TabularModel.load_from_checkpoint("examples/basic")
```
