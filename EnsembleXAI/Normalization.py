import numpy as np
import torch


def mean_var_normalize(explanation_tensor, delta=1e-15):
    var, mean = torch.var_mean(explanation_tensor, dim=[0, 2, 3, 4], unbiased=True, keepdim=True)
    if torch.min(var.abs()) < delta:
        raise ZeroDivisionError("Variance close to 0. Can't normalize")
    return (explanation_tensor - mean) / torch.sqrt(var)


def median_iqr_normalize(explanation_tensor, delta=1e-15):
    explanation_tensor = explanation_tensor.squeeze(0)
    median = torch.tensor(np.median(explanation_tensor, axis=[1, 2, 3])).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    q_25 = torch.tensor(np.quantile(explanation_tensor, q=0.25, axis=[1, 2, 3])).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    q_75 = torch.tensor(np.quantile(explanation_tensor, q=0.75, axis=[1, 2, 3])).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    iqr = q_75 - q_25
    if torch.min(iqr) < delta:
        raise ZeroDivisionError("IQR close to 0. Can't normalize")
    return (explanation_tensor - median) / iqr


def second_moment_normalize(explanation_tensor, eps=1e-25):
    std = torch.std(explanation_tensor, dim=[3, 4], keepdim=True)
    mean_std = torch.mean(std, dim=2, keepdim=True)
    normalized_tensor = explanation_tensor / (mean_std + eps)
    return normalized_tensor
