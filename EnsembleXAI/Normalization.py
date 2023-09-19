import numpy as np
import torch


def mean_var_normalize(explanation_tensor, eps=1e-25):
    var, mean = torch.var_mean(explanation_tensor, dim=[0, 2, 3, 4], unbiased=True, keepdim=True)
    return (explanation_tensor - mean) / torch.sqrt(var + eps)


def median_iqr_normalize(explanation_tensor, eps=1e-25):
    explanation_tensor = explanation_tensor.squeeze(0)
    median = torch.tensor(np.median(explanation_tensor, axis=[1, 2, 3])).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    q_25 = torch.tensor(np.quantile(explanation_tensor, q=0.25, axis=[1, 2, 3])).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    q_75 = torch.tensor(np.quantile(explanation_tensor, q=0.75, axis=[1, 2, 3])).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    iqr = q_75 - q_25
    return (explanation_tensor - median) / (iqr + eps)


def second_moment_normalize(explanation_tensor, eps=1e-25):
    std = torch.std(explanation_tensor, dim=[3, 4], keepdim=True)
    mean_std = torch.mean(std, dim=2, keepdim=True)
    normalized_tensor = explanation_tensor / (mean_std + eps)
    return normalized_tensor
