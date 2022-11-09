from torch import Tensor
from typing import TypeVar, Tuple, List, Callable

TensorOrTupleOfTensorsGeneric = TypeVar(
    "TensorOrTupleOfTensorsGeneric", Tensor, Tuple[Tensor, ...]
)


def aggregate(inputs: TensorOrTupleOfTensorsGeneric, aggregating_func: str) -> Tensor:
    pass


def ensemble(inputs: TensorOrTupleOfTensorsGeneric, metrics: List[Callable], weights: List[float]) -> Tensor:
    pass


def ensembleXAI(inputs: Tensor, masks: Tensor) -> Tensor:
    pass
