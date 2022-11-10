import torch

from copy import deepcopy
from torch import Tensor, stack
from typing import TypeVar, Tuple, List, Callable, Union

TensorOrTupleOfTensorsGeneric = TypeVar("TensorOrTupleOfTensorsGeneric", Tensor, Tuple[Tensor, ...])


def aggregate(inputs: TensorOrTupleOfTensorsGeneric, aggregating_func: Union[str, Callable]) -> Tensor:
    # input tensor dims: observations x explanations x single explanation
    if isinstance(aggregating_func, str):
        assert aggregating_func in ['avg', 'min', 'max']

    parsed_inputs = deepcopy(inputs)
    if isinstance(inputs, tuple):
        if inputs[0].dim() <= 3:
            # multiple observations with explanations as tensor
            parsed_inputs = stack(inputs)
    if parsed_inputs.dim() == 3:
        # single observation with multiple explanations
        parsed_inputs = parsed_inputs[None, :]

    input_size = parsed_inputs.size()
    n_explanations = input_size[1]
    new_size = (input_size[0], 1, input_size[2], input_size[3])

    if aggregating_func == 'avg':
        output = torch.squeeze(1 / n_explanations * parsed_inputs.sum_to_size(new_size), dim=1)

    if aggregating_func == 'max':
        output = parsed_inputs.amax(1)

    if aggregating_func == 'min':
        output = parsed_inputs.amin(1)

    return output


def ensemble(inputs: TensorOrTupleOfTensorsGeneric, metrics: List[Callable], weights: List[float]) -> Tensor:
    pass


def ensembleXAI(inputs: Tensor, masks: Tensor) -> Tensor:
    pass
