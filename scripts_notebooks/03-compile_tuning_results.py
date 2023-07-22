import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import optuna
from collections import namedtuple

Data = namedtuple('Data', ["type","name", "path", "config"])

DATA_PATH = Path("data")
OUTPUT_PATH = Path("output")
STUDY_PATH = Path("study")

# POL was run for classification and regression. So calling it out
POL_CLS_UUID = "66dc899c4c4f4e4f822d02bdddb18f80"

experiments = [
 'classification|default-of-credit-card-clients',
 'classification|heloc',
 'classification|eye_movements',
 'classification|Higgs',
 'classification|pol',
 'classification|albert',
 'classification|road-safety',
 'classification|MiniBooNE',
 'classification|covertype',
 'classification|jannis',
 'classification|Bioresponse',
 'regression|cpu_act',
 'regression|Ailerons',
 'regression|yprop_4_1',
 'regression|superconduct',
 'regression|Allstate_Claims_Severity',
 'regression|topo_2_1',
 'regression|Mercedes_Benz_Greener_Manufacturing',
]

datasets = []
for experiment in experiments:
    typ, name = experiment.split("|")
    folder = DATA_PATH/name
    config_files = list(folder.glob("*config*"))
    config = np.load(config_files[0], allow_pickle=True).item()
    config["n_iter"] = len(config_files)
    datasets.append(Data(typ, name, folder, config))
dataset_dict = {d.name: d for d in datasets}
dataset_names = list(dataset_dict.keys())

#Compiling and selection subset from benchmark results
benchmark = pd.read_csv('data/benchmark_total.csv', dtype={"_timestamp": "str"})

#convert _timestamp to int64, handling empty strings
def convert_timestamp(x):
    try:
        return int(x)
    except:
        return 0
benchmark["_timestamp"] = benchmark["_timestamp"].apply(convert_timestamp)

# Selecting only openml_no_transform
benchmark  = benchmark .loc[benchmark.data__method_name == "openml_no_transform"]
benchmark .drop(columns=['data__method_name'], inplace=True)
# Selecting only transformed_target == 0
benchmark  = benchmark .loc[benchmark.transformed_target == 0]
benchmark .drop(columns=['transformed_target'], inplace=True)
# selecting only hp="random"
benchmark  = benchmark .loc[benchmark.hp == "random"]
benchmark .drop(columns=['hp'], inplace=True)

sel_columns = [
 'Unnamed: 0',
 'benchmark',
 'data__keyword',
 '_timestamp',
 'model_name',
 'model_type',
 'one_hot_encoder',

 'mean_time',
 'std_time',

 'mean_test_score',

 'max_test_score',
 'min_test_score',
 'std_test_score',

 'test_scores',
 'times',
 ]
benchmark = benchmark.loc[:, sel_columns]

benchmark_l = []
for d in datasets:
    name_mask = benchmark.data__keyword == d.name
    type_mask = benchmark.benchmark.str.contains(d.type)
    cat_mask = benchmark.benchmark.str.contains("categorical") if d.config["data__categorical"] else benchmark.benchmark.str.contains("numerical")
    df = benchmark.loc[name_mask & type_mask & cat_mask]
    benchmark_l.append(df)
benchmark = pd.concat(benchmark_l)
benchmark['mean_test_score'] = pd.to_numeric(benchmark['mean_test_score'], errors="coerce")
benchmark.to_parquet("data/benchmark_total.parquet")

def calc_best_row(df):
    if df.shape[0] == 0:
        return df
    best_idx = df.mean_test_score.idxmax()
    best_row = df.loc[best_idx]
    best_row["best"] = True
    return best_row
benchmark = benchmark.groupby(["benchmark", "data__keyword", "model_name"]).apply(calc_best_row)

benchmark = benchmark.loc[:, ["mean_time", "std_time", "mean_test_score", "max_test_score", "min_test_score", "std_test_score", "test_scores", "times"]]
benchmark.to_parquet("data/benchmark_best_rows.parquet")

# Compiling Tuning Results
sel_columns = [
 'Unnamed: 0',
 'benchmark',
 'data__keyword',
 '_timestamp',
 'model_name',
 'model_type',
 'one_hot_encoder',

 'mean_time',
 'std_time',

 'mean_test_score',

 'max_test_score',
 'min_test_score',
 'std_test_score',

 'test_scores',
 'times',
 ]


experiment_results = pd.DataFrame(columns=sel_columns)


def get_study(dataset_name):
    if dataset_name == "pol":
        return pd.read_csv(OUTPUT_PATH/f"study_pol_{POL_CLS_UUID}.csv")
    folder = OUTPUT_PATH
    outputs = [f for f in folder.glob("study*") if dataset_name in f.name]
    if len(outputs)==1:
        return pd.read_csv(outputs[0])
    else: # For cncelled and unfinished trials
        folder  = STUDY_PATH
        study_path = [f for f in folder.glob(f"*{dataset_name}*.db") if "full" not in f.name][0]
        study_name = study_path.name.split(".")[0]
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{study_path}")
        return study.trials_dataframe()
    
def format_study(study, dataset):
    type = dataset_dict[dataset.name].type
    categorical = dataset_dict[dataset.name].config["data__categorical"]
    if categorical==1.0:
        benchmark = "categorical"
    else:
        benchmark = "numerical"
    benchmark+=f"_{type}_medium"
    study['benchmark'] = benchmark
    study["data__keyword"] = dataset.name
    study["duration"] = pd.to_timedelta(study["duration"], errors="coerce")
    study["mean_time"] = study["duration"].dt.total_seconds()/dataset_dict[dataset.name].config["n_iter"]
    rename_dict = {
        "datetime_start": "_timestamp",
        "value": "mean_test_score",
    }
    study.rename(columns=rename_dict, inplace=True)
    study["model_name"] = "GANDALF"
    study["model_type"] = "PyTorchTabular"
    param_cols = [c for c in study.columns if c.startswith("params_")]
    study['params']=study[param_cols].apply(dict, axis=1)
    study = study.loc[study.state=="COMPLETE"]
    intersection_cols = list(set(study.columns).intersection(set(sel_columns)))
    study = study.loc[:, intersection_cols+["params"]]
    return study
            


#Add row for each experiment
for dataset in datasets:
    print(dataset.name)
    study = get_study(dataset.name)
    study = format_study(study, dataset)
    experiment_results = pd.concat([experiment_results,study], ignore_index=True)


experiment_results["_timestamp"] = pd.to_datetime(experiment_results["_timestamp"])

# ## Results Compilation


def calc_best_row(df):
    if df.shape[0] == 0:
        return df
    best_idx = df.mean_test_score.idxmax()
    best_row = df.loc[best_idx]
    best_row["best"] = True
    return best_row
experiment_results = experiment_results.groupby(["benchmark", "data__keyword", "model_name"]).apply(calc_best_row)


experiment_results = experiment_results.loc[:, ["mean_time", "mean_test_score", "test_scores", "times", "params"]]



benchmark_results = pd.read_parquet("data/benchmark_best_rows.parquet")


experiment_results = pd.concat([experiment_results, benchmark_results]).sort_values(["benchmark", "data__keyword", "model_name"])


experiment_results.to_parquet("data/tuning_results_best_rows.parquet")



