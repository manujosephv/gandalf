import numpy as np
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal


def gaussienize(x_train, x_val, x_test, y_train, y_val, y_test, type="standard", rng=None):
    """
    Gaussienize the data
    :param x: data to transform
    :param type: {"standard","robust", "quantile", "power", "quantile_uniform"}
    :return: the transformed data
    """
    print("Gaussienizing")
    if type == "identity":
        return x_train, x_val, x_test, y_train, y_val, y_test
    if type == "standard":
        t = StandardScaler()
    elif type == "robust":
        t = RobustScaler()
    elif type == "quantile":
        t = QuantileTransformer(output_distribution="normal", random_state=rng)
    elif type == "quantile_uniform":
        t = QuantileTransformer(output_distribution="uniform", random_state=rng)
    elif type == "power":
        t = PowerTransformer(random_state=rng)

    x_train = t.fit_transform(x_train)
    x_val = t.transform(x_val)
    x_test = t.transform(x_test)

    return x_train, x_val, x_test, y_train, y_val, y_test


def balance(x_train, x_test, y_train, y_test, rng):
    indices_train = [(y_train == 0), (y_train == 1)]
    max_class = np.argmax(list(map(sum, indices_train)))
    min_class = np.argmin(list(map(sum, indices_train)))
    n_samples_min_class = sum(indices_train[min_class])
    indices_max_class = rng.choice(np.where(indices_train[max_class])[0], n_samples_min_class, replace=False)
    indices_min_class = np.where(indices_train[min_class])[0]
    total_indices_train = np.concatenate((indices_max_class, indices_min_class))

    indices_test = [(y_train == 0), (y_train == 1)]
    max_class = np.argmax(list(map(sum, indices_test)))
    min_class = np.argmin(list(map(sum, indices_test)))
    n_samples_min_class = sum(indices_test[min_class])
    indices_max_class = rng.choice(np.where(indices_test[max_class])[0], n_samples_min_class, replace=False)
    indices_min_class = np.where(indices_test[min_class])[0]
    total_indices_test = np.concatenate((indices_max_class, indices_min_class))
    return x_train[total_indices_train], x_test[total_indices_test], y_train[total_indices_train], y_test[
        total_indices_test]


def limit_size(x, y, n_samples, rng):
    indices = list(range(len(y)))
    chosen_indices = rng.choice(indices, n_samples, replace=False)
    return x[chosen_indices], y[chosen_indices]
