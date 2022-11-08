from torch import Tensor
from captum.attr import Attribution
from torch.utils.data import Dataset
from typing import Callable, Union


def consistency(explanations: list[Tensor]) -> Tensor:
    """
    Opis: Mierzy jak bardzo wyjaśnienia różnych modeli uczenia maszynowego są do siebie podobne.
    Argumenty wejściowe: Lista<Tensor> (lista wyjaśnień które chcemy ze sobą porównać)
    Wartości wyjściowe: Tensor (wynik metryki)
    :return:
    """
    pass


def stability(explanator: Attribution, image: Tensor) -> Tensor:
    """
    Opis: Mierzy jak podobne wyjaśnienia otrzymamy dla podobnych danych wejściowych.
    Argumenty wejściowe: obiekt ‘callable’ (metoda zwracająca wyjaśnienie), tensor (obrazek który chcemy wyjaśnić)
    Wartości wyjściowe: Tensor (wynik metryki)
    :return:
    """
    pass


def decision_impact_ratio(dataset: Dataset, predictor: Callable[..., Tensor], explanator: Attribution,
                          baseline: Tensor) -> Tensor:
    """
    Czy chcemy tylko na tescie to robic?
    Opis: Jest to odsetek obserwacji, dla których po usunięciu obszaru wrażliwości (wskazanego przez wyjaśnienie)
    klasyfikacja modelu zmieniła się.
    Argumenty wejściowe: dataset, forward z modelu, funkcja wyjasnienia, baseline do podmiany pixeli
    Wartości wyjściowe: Tensor (wynik metryki)
    :return:
    """
    pass


def confidence_impact_ratio(dataset: Dataset, probs: Callable[..., Tensor], explanator: Attribution,
                            baseline: Tensor) -> Tensor:
    """
    Opis: Średni spadek estymowanego prawdopodobieństwa klasyfikacji po zasłonięciu obszaru wrażliwości.
    Argumenty wejściowe: dataset, funkcja na probsy z modelu, funkcja wyjasnienia, baseline do podmiany pixeli
    Wartości wyjściowe: Tensor (wynik metryki)
    :return:
    """
    pass


def accordance_recall(mask: Tensor, explanation: Tensor, threshold: float = 0.5) -> Tensor:
    """
    Opis: Mierzy jaką część maski wykryło wyjaśnienie.
    Argumenty wejściowe: Tensor (maska obrazu), Tensor (wyjaśnienie/obszar krytyczny), Tensor/skalar (opcjonalnie,
    jaka minimalna wartość w wyjaśnieniu ma być uznana za ważną)
    Wartości wyjściowe: Tensor (wynik metryki)
    :return:
    """
    pass


def accordance_precision(mask: Tensor, explanation: Tensor, threshold: float = 0.5) -> Tensor:
    """
    Opis: mierzy jaką część wyjaśnienia stanowiła maska.
    Argumenty wejściowe: Tensor (maska obrazu), Tensor (wyjasnienie/obszar krytyczny), Tensor/skalar (opcjonalnie,
    jaka minimalna wartość w wyjaśnieniu ma być uznana za ważną)
    Wartości wyjściowe: Tensor (wynik metryki)
    :return:
    """
    pass


def F1_score(mask: Tensor, explanation: Tensor, threshold: float = 0.5) -> Tensor:
    """
    Opis: Średnia harmoniczna Accordance recall i Accordance precision.
    Argumenty wejściowe: Tensor (maska obrazu), Tensor (wyjasnienie/obszar krytyczny), Tensor/skalar (opcjonalnie,
    jaka minimalna wartość w wyjaśnieniu ma być uznana za ważną)
    Wartości wyjściowe: Tensor (wynik metryki)
    :return:
    """
    pass


def intersection_over_union(mask: Tensor, explanation: Tensor, threshold: float = 0.5) -> Tensor:
    """
    Opis: Pole iloczynu maski i wyjaśnienia podzielone przez pole sumy maski i wyjaśnienia.
    Argumenty wejściowe: Tensor (maska obrazu), Tensor (wyjasnienie/obszar krytyczny), Tensor/skalar (opcjonalnie,
    jaka minimalna wartość w wyjaśnieniu ma być uznana za ważną)
    Wartości wyjściowe: Tensor (wynik metryki)
    :return:
    """
    pass


def ensemble_score(weights: Union[list, Tensor], explanations: Union[list[Tensor], Tensor]) -> Tensor:
    """
    Opis: średnia ważona innych metryk. Ensemble_score(wagi, metryki) -> tensor
    Argumenty wejściowe: Tensor/Lista (lista z wagami dla poszczególnych metryk), Tensor/Lista (lista z wynikami
    poszczególnych metryk)
    Wartości wyjściowe: Tensor (wynik metryki)
    :return:
    """
    pass
