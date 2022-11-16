from captum.attr import Attribution
from torch.utils.data import Dataset
from typing import Callable, Union, List
import itertools
import torch


def replace_mask(image: torch.Tensor, mask: torch.Tensor, value: Union[int, float]) -> torch.Tensor:
    temp_image = torch.clone(image)
    reshaped_mask = mask.repeat(3, 1, 1)
    temp_image[reshaped_mask] = value
    return temp_image


def _matrix_norm_2(exp1: torch.Tensor, exp2: torch.Tensor) -> torch.Tensor:
    difference = (exp1 - exp2).float()
    return torch.linalg.matrix_norm(difference, ord=2)


def _intersection(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    threshold1: float = 0.0,
    threshold2: float = 0.0,
) -> torch.Tensor:
    return torch.logical_and(
        torch.abs(tensor1) > threshold1, torch.abs(tensor2) > threshold2
    )


def _union(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    threshold1: float = 0.0,
    threshold2: float = 0.0,
) -> torch.Tensor:
    return torch.logical_or(
        torch.abs(tensor1) > threshold1, torch.abs(tensor2) > threshold2
    )


def consistency(explanations: List[torch.Tensor]) -> torch.Tensor:
    """
    Opis: Mierzy jak bardzo wyjaśnienia różnych modeli uczenia maszynowego są do siebie podobne.
    Argumenty wejściowe: Lista<torch.Tensor> (lista wyjaśnień które chcemy ze sobą porównać)
    Wartości wyjściowe: torch.Tensor (wynik metryki)
    :return:
    C(phi,...) = [max_{a,b}(||phi_{j}^{e->m_a} - phi_{j}^{e->m_b}||_2) + 1]^{-1}, phi_j-wyjasnienie j tego zdjęcia lub
    [max_{a,b}(||phi_{j}^{e_a->m} - phi_{j}^{e_b->m}||_2) + 1]^{-1}
    """
    diffs = [
        _matrix_norm_2(exp1, exp2)
        for exp1, exp2 in itertools.combinations(explanations, 2)
    ]
    return 1 / max(diffs) + 1


def stability(
    explanator: Attribution, model, image: torch.Tensor, epsilon: float = 0.1
) -> torch.Tensor:
    """
    Opis: Mierzy jak podobne wyjaśnienia otrzymamy dla podobnych danych wejściowych.
    Argumenty wejściowe: obiekt ‘callable’ (metoda zwracająca wyjaśnienie), torch.tensor (obrazek który chcemy wyjaśnić)
    Wartości wyjściowe: torch.Tensor (wynik metryki)
    :return:
    L(phi, X) = max_{x_j} (||x_i-x_j||_{2}/(||phi_i^{e->m} - phi_j^{e->m}||_{2}+1))
    https://github.com/sbobek/inxai/blob/main/inxai/global_metrics.py
    """
    pass


def decision_impact_ratio(
    dataset: Dataset,
    predictor: Callable[..., torch.Tensor],
    explanator: Attribution,
    baseline: torch.Tensor,
) -> torch.Tensor:
    """
    Czy chcemy tylko na tescie to robic?
    Opis: Jest to odsetek obserwacji, dla których po usunięciu obszaru wrażliwości (wskazanego przez wyjaśnienie)
    klasyfikacja modelu zmieniła się.
    Argumenty wejściowe: dataset, forward z modelu, funkcja wyjasnienia, baseline do podmiany pixeli
    Wartości wyjściowe: torch.Tensor (wynik metryki)
    :return:
    DIR = Suma po i (1 jeżeli D(x_i)=/=D(x_i-c_i) else 0)/N, D to klasyfikacja, c_i obszar krytyczny
    """
    pass


def confidence_impact_ratio(
    dataset: Dataset,
    probs: Callable[..., torch.Tensor],
    explanator: Attribution,
    baseline: torch.Tensor,
) -> torch.Tensor:
    """
    Opis: Średni spadek estymowanego prawdopodobieństwa klasyfikacji po zasłonięciu obszaru wrażliwości.
    Argumenty wejściowe: dataset, funkcja na probsy z modelu, funkcja wyjasnienia, baseline do podmiany pixeli
    Wartości wyjściowe: torch.Tensor (wynik metryki)
    :return:
    CIR = Suma po i max(C(x_i)-C(x_i-c_i), 0)/N , C to probabilities, c_i obszar krytyczny
    """
    pass


def accordance_recall(
    mask: torch.Tensor, explanation: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """
        Opis: Mierzy jaką część maski wykryło wyjaśnienie.
        Argumenty wejściowe: torch.Tensor (maska obrazu), torch.Tensor (wyjaśnienie/obszar krytyczny),
        torch.Tensor/skalar (opcjonalnie,
        jaka minimalna wartość w wyjaśnieniu ma być uznana za ważną)
        Wartości wyjściowe: torch.Tensor (wynik metryki)
        :return:
        Let S(𝑥) be the suspicious pneumonia area that is annotated
    by the clinician for image x and let F(𝑥) be the critical area that
    is identified by the interpretation method
        recall_i=(S(x_i) czesc wspolna F(x_i))/S(x_i)
        recall = sum_i(recall_i)/N
    """
    # maska logiczna, jest czy nie jest w masce
    overlaping_area = _intersection(explanation, mask, threshold1=threshold)
    return torch.sum(overlaping_area) / torch.sum(torch.abs(mask) != 0)


def accordance_precision(
    mask: torch.Tensor, explanation: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """
        Opis: mierzy jaką część wyjaśnienia stanowiła maska.
        Argumenty wejściowe: torch.Tensor (maska obrazu), torch.Tensor (wyjasnienie/obszar krytyczny),
        torch.Tensor/skalar (opcjonalnie,
        jaka minimalna wartość w wyjaśnieniu ma być uznana za ważną)
        Wartości wyjściowe: torch.Tensor (wynik metryki)
        :return:
        Let S(𝑥) be the suspicious pneumonia area that is annotated
    by the clinician for image x and let F(𝑥) be the critical area that
    is identified by the interpretation method
        precision_i=(S(x_i) czesc wspolna F(x_i))/F(x_i)
        precision = sum_i (precision_i)/N
    """
    # maska logiczna, jest czy nie jest w masce
    overlaping_area = _intersection(explanation, mask, threshold1=threshold)
    return torch.sum(overlaping_area) / torch.sum(torch.abs(explanation) > threshold)


def set_accordance_recall(masks, explanations, threshold=0.5):
    # tensor with a scalar value is expected as a result of accordance_recall
    recalls = [
        accordance_recall(mask, expl, threshold=threshold).item()
        for mask, expl in zip(masks, explanations)
    ]
    return recalls


def set_accordance_precision(masks, explanations, threshold=0.5):
    # tensor with a scalar value is expected as a result of accordance_precision
    precs = [
        accordance_precision(mask, expl, threshold=threshold).item()
        for mask, expl in zip(masks, explanations)
    ]
    return precs


def F1_score(
    masks: List[torch.Tensor], explanations: List[torch.Tensor], threshold: float = 0.5
) -> torch.Tensor:
    """
        Opis: Średnia harmoniczna Accordance recall i Accordance precision.
        Argumenty wejściowe: torch.Tensor (maska obrazu), torch.Tensor (wyjasnienie/obszar krytyczny),
        torch.Tensor/skalar (opcjonalnie,
        jaka minimalna wartość w wyjaśnieniu ma być uznana za ważną)
        Wartości wyjściowe: torch.Tensor (wynik metryki)
        :return:
        Let S(𝑥) be the suspicious pneumonia area that is annotated
    by the clinician for image x and let F(𝑥) be the critical area that
    is identified by the interpretation method
        Accordance F1 = 1/N * sum_i (2*(recall_i + precision_i)/(recall_i*precision_i))
    """
    acc_recall = set_accordance_recall(masks, explanations, threshold=threshold)
    acc_prec = set_accordance_precision(masks, explanations, threshold=threshold)
    values = [
        2 * (recall * prec) / (recall + prec)
        for (recall, prec) in zip(acc_recall, acc_prec)
    ]
    return sum(values) / len(values)


def intersection_over_union(
    masks: List[torch.Tensor], explanations: List[torch.Tensor], threshold: float = 0.5
) -> torch.Tensor:
    """
        Opis: Pole iloczynu maski i wyjaśnienia podzielone przez pole sumy maski i wyjaśnienia.
        Argumenty wejściowe: torch.Tensor (maska obrazu), torch.Tensor (wyjasnienie/obszar krytyczny),
        torch.Tensor/skalar (opcjonalnie,
        jaka minimalna wartość w wyjaśnieniu ma być uznana za ważną)
        Wartości wyjściowe: torch.Tensor (wynik metryki)
        :return:
        Let S(𝑥) be the suspicious pneumonia area that is annotated
    by the clinician for image x and let F(𝑥) be the critical area that
    is identified by the interpretation method
        IOU=1/N * sum_i(S(x_i) cz. wspolna F(x_i)/S(x_i) suma F(x_i))
    """
    values = [
        torch.sum(_intersection(expl, mask, threshold1=threshold))
        / torch.sum(_union(expl, mask, threshold1=threshold))
        for (expl, mask) in zip(explanations, masks)
    ]
    return sum(values) / len(values)


def ensemble_score(
    weights: Union[List, torch.Tensor],
    metrics_scores: Union[List[torch.Tensor], torch.Tensor, List[float]],
) -> torch.Tensor:
    """
    Opis: średnia ważona innych metryk. Ensemble_score(wagi, metryki) -> torch.tensor
    Argumenty wejściowe: torch.Tensor/Lista (lista z wagami dla poszczególnych metryk), torch.Tensor/Lista
    (lista z wynikami poszczególnych metryk)
    Wartości wyjściowe: torch.Tensor (wynik metryki)
    :return:
    ES(w, M) = sum_{i}(w_i * M_i)
    """
    return sum(
        [
            weight * metric_score
            for (weight, metric_score) in zip(weights, metrics_scores)
        ]
    )
