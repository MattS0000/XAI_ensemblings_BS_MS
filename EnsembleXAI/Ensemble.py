import torch

from copy import deepcopy
from EnsembleXAI.Metrics import ensemble_score
import numpy as np
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from torch import Tensor, stack
from typing import TypeVar, Tuple, List, Callable, Union

TensorOrTupleOfTensorsGeneric = TypeVar("TensorOrTupleOfTensorsGeneric", Tensor, Tuple[Tensor, ...])


def _apply_over_axis(x: torch.Tensor, function: Callable, axis: int) -> torch.Tensor:
    return torch.stack([
        function(x_i) for i, x_i in enumerate(torch.unbind(x, dim=axis), 0)
    ], dim=axis)


def _reformat_input_tensors(inputs: TensorOrTupleOfTensorsGeneric) -> Tensor:
    # change tuple of tensors into one standard 5d tensor
    # with dim:
    # 0 - observations
    # 1 - different explanations for one observation
    # 2 - channels (colors) of one explanation
    # 3 - height of channel
    # 4 - width of channel
    parsed_inputs = deepcopy(inputs)
    if isinstance(inputs, tuple) or isinstance(inputs, list):
        if inputs[0].dim() <= 4:
            # multiple observations with explanations as tensor
            parsed_inputs = stack(inputs)
    if parsed_inputs.dim() == 3:
        parsed_inputs = parsed_inputs[None, :]
    if parsed_inputs.dim() == 4:
        # single observation with multiple explanations
        parsed_inputs = parsed_inputs[None, :]

    return parsed_inputs


# basic
def aggregate(inputs: TensorOrTupleOfTensorsGeneric,
              aggregating_func: Union[str, Callable[[Tensor], Tensor]]) -> Tensor:
    """
    Aggregate explanations in the simplest way.

    Use provided aggregating functions or pass a custom callable. Combine explanations for every observation and get
    one aggregated explanation for every observation.


    Parameters
    ----------
    inputs : TensorOrTupleOfTensorsGeneric
        Explanations in form of tuple of tensors or tensor. `inputs` dimensions correspond to no. of observations,
        no. of explanations for each observation, and single explanation.
    aggregating_func : Union[str, Callable[[Tensor], Tensor]]
        Aggregating function. Can be string, one of 'avg', 'min', 'max',
        or a function from a list of tensors to tensor.

    Returns
    -------
    Tensor
        Aggregated explanations. Dimensions correspond to no. of observations, aggregated explanation.

    See Also
    --------
    ensemble : Aggregation weighted by quality of each explanation.
    ensembleXAI : Use Kernel Ridge Regression for aggregation, suitable when masks are available.

    Examples
    --------
    TODO
    """
    # input tensor dims: observations x explanations x single explanation
    assert isinstance(aggregating_func, str) or isinstance(aggregating_func, Callable)

    if isinstance(aggregating_func, str):
        assert aggregating_func in ['avg', 'min', 'max']

    parsed_inputs = _reformat_input_tensors(inputs)

    input_size = parsed_inputs.size()
    n_explanations = input_size[1]
    new_size = (input_size[0], 1, input_size[2], input_size[3], input_size[4])

    if aggregating_func == 'avg':
        output = torch.squeeze(1 / n_explanations * parsed_inputs.sum_to_size(new_size), dim=1)

    if aggregating_func == 'max':
        output = parsed_inputs.amax(1)

    if aggregating_func == 'min':
        output = parsed_inputs.amin(1)

    if isinstance(aggregating_func, Callable):
        output = _apply_over_axis(parsed_inputs, aggregating_func, 0)

    return output


def _normalize_across_dataset(parsed_inputs, delta=0.00001):
    # mean, std normalization
    var, mean = torch.var_mean(parsed_inputs, dim=[0, 2, 3, 4], unbiased=True)
    if torch.min(var.abs()) < delta:
        raise ZeroDivisionError("Variance close to 0. Can't normalize")
    return (parsed_inputs - mean) / torch.sqrt(var)


# autoweighted
def ensemble(inputs: TensorOrTupleOfTensorsGeneric, metrics: List[Callable], weights: List[float]) -> Tensor:
    """
    Aggregate explanations weighted by their quality measured by metrics.

    This function in an implementation of explanation ensemble algorithm published in [1]_. It uses
    :func:`EnsembleXAI.Metrics.ensemble_score` to calculate quality of each explanation.

    Parameters
    ----------
    inputs : TensorOrTupleOfTensorsGeneric
        Explanations in form of tuple of tensors or tensor. `inputs` dimensions correspond to no. of observations,
        no. of explanations for each observation, and single explanation.
    metrics : List[Callable]
        Metrics used to assess the quality of an explanation.
    weights : List[float]
        Weights used to calculate :func:`EnsembleXAI.Metrics.ensemble_score` of every explanation.
    Returns
    -------
    Tensor
        Weighted arithmetic mean of explanations, weighted by :func:`EnsembleXAI.Metrics.ensemble_score`.
        Dimensions correspond to no. of observations, aggregated explanation.

    See Also
    --------
    aggregate : Simple aggregation by function, like average.
    ensembleXAI : Use Kernel Ridge Regression for aggregation, suitable when masks are available.

    Notes
    -----
    Explanations are normalized by mean and standard deviation before aggregation to ensure comparable values.

    References
    ----------
    .. [1] Bobek, S., Bałaga, P., Nalepa, G.J. (2021), "Towards Model-Agnostic Ensemble Explanations."
        In: Paszynski, M., Kranzlmüller, D., Krzhizhanovskaya, V.V., Dongarra, J.J., Sloot, P.M. (eds)
        Computational Science – ICCS 2021. ICCS 2021. Lecture Notes in Computer Science(), vol 12745. Springer,
        Cham. https://doi.org/10.1007/978-3-030-77970-2_4

    Examples
    --------
    TODO

    """
    parsed_inputs = _reformat_input_tensors(inputs)

    # calculate metrics and ensemble scores
    metric_vals = [[metric(explanation) for metric in metrics] for explanation in torch.unbind(parsed_inputs, dim=1)]
    ensemble_scores = torch.stack([ensemble_score(weights, metric_val) for metric_val in metric_vals])
    ensemble_scores.transpose_(0, 1)

    normalized_exp = _normalize_across_dataset(parsed_inputs)

    # allocate array for ensemble explanations
    n = parsed_inputs.size()[0]
    results = [0] * n

    for observation, scores, i in zip(torch.unbind(normalized_exp), torch.unbind(ensemble_scores), range(n)):
        # multiply single explanation by its ensemble score
        weighted_exp = torch.stack([
            exp * score for exp, score in zip(torch.unbind(observation), torch.unbind(scores))
        ])
        # sum weighted explanations and normalize by sum of scores
        ensemble_exp = torch.sum(weighted_exp, dim=0) / scores.sum()
        results[i] = ensemble_exp

    return torch.stack(results)


# supervisedXAI
def ensembleXAI(inputs: TensorOrTupleOfTensorsGeneric, masks: TensorOrTupleOfTensorsGeneric, n_folds: int = 3,
                shuffle=False, random_state=None) -> Tensor:
    """
    Aggregate explanations by training supervised machine learning model.

    This function in an implementation of explanation ensemble algorithm published in [1]_. It uses
    :class:`sklearn.kernel_ridge.KernelRidge` to train the Kernel Ridge Regression (KRR) model with explanations
    as inputs :math:`X` and masks as output :math:`y`.
    K-Fold split is used to generate aggregated explanations without information leakage. Internally uses
    :class:`sklearn.model_selection.KFold` to make the split.


    Parameters
    ----------
    inputs : TensorOrTupleOfTensorsGeneric
        Explanations in form of tuple of tensors or tensor. `inputs` dimensions correspond to no. of observations,
        no. of explanations for each observation, and single explanation.
    masks : TensorOrTupleOfTensorsGeneric
        Masks used by KRR model as output. Should be the same shape as `inputs`.
    n_folds : int, default 3
        Number of folds used to train the KRR model. `n_folds` should be an `int` greater than 1. When `n_folds` is
        equal to no. of observations in `inputs`, "leave one out" training is done.
    shuffle: Any, default False
        If `True` inputs and masks will be shuffled before k-fold split. Internally passed
        to :class:`sklearn.model_selection.KFold`.
    random_state: Any, default None
        Used only when `shuffle` is `True`. Internally passed to :class:`sklearn.model_selection.KFold`.
    Returns
    -------
    Tensor
        Tensor of KRR model outputs, which are the aggregated explanations.

    See Also
    --------
    aggregate : Simple aggregation by function, like average.
    ensemble : Aggregation weighted by quality of each explanation.

    References
    ----------
    .. [1] L. Zou et al., "Ensemble image explainable AI (XAI) algorithm for severe community-acquired
        pneumonia and COVID-19 respiratory infections,"
        in IEEE Transactions on Artificial Intelligence, doi: 10.1109/TAI.2022.3153754.

    Examples
    --------
    TODO
    """

    assert len(inputs) == len(masks)
    assert n_folds > 1
    # reshape do 1d array for each observation
    parsed_inputs = _reformat_input_tensors(inputs)
    input_shape = parsed_inputs.shape
    parsed_inputs = parsed_inputs.numpy().reshape((len(inputs), -1))
    labels = _reformat_input_tensors(masks).squeeze().numpy().reshape((len(inputs), -1))

    kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    ensembled = [0] * n_folds
    indices = np.empty(1, dtype=int)

    for idx, (train_index, test_index) in enumerate(kf.split(parsed_inputs, labels)):
        # get observations split by k-fold
        X_train, X_test = (parsed_inputs[train_index]), (parsed_inputs[test_index])
        y_train = labels[train_index]
        # train KRR
        krr = KernelRidge(
            alpha=1,  # regularization
            kernel='polynomial'  # choose one from:
            # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/pairwise.py#L2050
        )
        krr.fit(X_train, y_train)
        # predict masks for observations currently in test group
        y_predicted = krr.predict(X_test)
        # reshape predictions and save them and indices to recreate original order later
        ensembled[idx] = np.concatenate([y_predicted] * 3).reshape((tuple([len(X_test)]) + input_shape[2:5]))
        indices = np.concatenate([indices, test_index])

    # sort output to match input order
    indices = indices[1:]
    ensembled = np.concatenate(ensembled)
    ensembled_ind = indices.argsort()
    ensembled = ensembled[ensembled_ind[::1]]

    return torch.from_numpy(ensembled)
