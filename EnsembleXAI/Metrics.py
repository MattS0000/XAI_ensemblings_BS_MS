from typing import Callable, Union, List, Tuple
import itertools
import torch


def replace_masks(
    images: torch.Tensor, masks: torch.Tensor, value: Union[int, float] = 0
) -> torch.Tensor:
    """
    Replaces values in images where masks exist.

    Replaces data in the images Tensor with one value in the spots where masks Tensor is True.

    Parameters
    ----------
    images: torch.Tensor
        4D Tensor of the images with shape (number of photos, RGB channel, height, width)
    masks: torch.Tensor
        3D torch Boolean Tensor of the masks where true corresponds to mask present with shape (num of photos, height, width)
    value: int or float
        Value to use for replacing the data with

    Returns
    -------
    torch.Tensor
        4D Tensor, copy of images with the replaced data

    See Also
    --------
    replacetext : function description.

    Examples
    --------
    >>> replace_masks(x, y)
    answer
    """
    temp_images = torch.clone(images)
    reshaped_masks = masks.unsqueeze(dim=1).repeat(1, 3, 1, 1)
    temp_images[reshaped_masks] = value
    return temp_images


def tensor_to_list_tensors(tensors: torch.Tensor, depth: int) -> List[torch.Tensor]:
    """
    Splits first n dimensions of a Tensor into a list of Tensors.

    Splits the first n Tensor dimensions into a list of Tensors of length equal to product of the split dimensions sizes.

    Parameters
    ----------
    tensors: torch.Tensor
        Tensor to be split into a list.
    depth: int
        Value representing the depth to which to split the tensors, starting from the first dimension.
        Therefore, depth=1 represents splitting only the first dimension. Thus depth cannot be larger than the length of the Tensors shape.

    Returns
    -------
    list of torch.Tensor
        A single list consisting of all the split Tensors.

    See Also
    --------
    replacetext : function description.

    Examples
    --------
    >>> function(x, y)
    answer
    """
    tensor_list = [
        x.squeeze() for x in torch.tensor_split(tensors, tensors.shape[0], dim=0)
    ]
    for i in range(depth - 1):
        tensor_list = [
            y.squeeze()
            for x in tensor_list
            for y in torch.tensor_split(x, x.shape[0], dim=0)
        ]
    return tensor_list


def _matrix_norm_2(
    matrix1: torch.Tensor, matrix2: torch.Tensor, sum_dim: int = None
) -> torch.Tensor:
    """
    Computes the 2-norm of two matrices.

    Computes the 2-norm of two matrices. By default works on the last two dimensions of the Tensor,
    which can be extended by the sum_dim parameter to one of the left dimensions of the Tensor.

    Parameters
    ----------
    matrix1: torch.Tensor
        Tensor with one of the matrices to compute the norm.
    matrix2: torch.Tensor
        Tensor with the second of the matrices to compute the norm.
    sum_dim: int
        Optional dimension to extend the calculation to

    Returns
    -------
    torch.Tensor
        Tensor with value or values of the 2-norm. The shape is same as both of the input matrices,
        except for last two removed dimensions and the optional dimension specified in sum_dim parameter.

    See Also
    --------
    replacetext : function description.

    Examples
    --------
    >>> function(x, y)
    answer
    """
    difference = (matrix1 - matrix2).float()
    norm = torch.linalg.matrix_norm(difference, ord=2)
    if sum_dim:
        norm = torch.pow(norm, 2)
        norm = torch.sum(norm, dim=sum_dim)
        norm = torch.sqrt(norm)
    return norm


def _intersection_mask(
    tensor1: torch.Tensor, tensor2: torch.Tensor,
    threshold1: float = 0.0, threshold2: float = 0.0,
) -> torch.Tensor:
    """
    Calculates the intersection of two masks.

    Calculates the logical 'and' intersections of two n-dimensional masks where the absolute value of data is greater than the thresholds.

    Parameters
    ----------
    tensor1: torch.Tensor
        First of the two masks.
    tensor2: torch.Tensor
        Second of the two masks.
    threshold1: float
        Threshold value for the first mask.
    threshold2: float
        Threshold value for the second mask.

    Returns
    -------
    torch.Tensor
        Boolean Tensor with True values where the masks intersect with value over thresholds.

    See Also
    --------
    replacetext : function description.

    Examples
    --------
    >>> function(x, y)
    answer
    """
    logical_mask = torch.logical_and(
        torch.abs(tensor1) > threshold1, torch.abs(tensor2) > threshold2
    )
    return logical_mask


def _union_mask(
    tensor1: torch.Tensor, tensor2: torch.Tensor,
    threshold1: float = 0.0, threshold2: float = 0.0,
) -> torch.Tensor:
    """
    Calculates the union of two masks.

    Calculates the logical 'or' union of two n-dimensional masks where the absolute value of data is greater than the thresholds.

    Parameters
    ----------
    tensor1: torch.Tensor
        First of the two masks.
    tensor2: torch.Tensor
        Second of the two masks.
    threshold1: float
        Threshold value for the first mask.
    threshold2: float
        Threshold value for the second mask.

    Returns
    -------
    torch.Tensor
        Boolean Tensor with True values on the union of the masks, where values are over thresholds.

    See Also
    --------
    replacetext : function description.

    Examples
    --------
    >>> function(x, y)
    answer
    """
    logical_mask = torch.logical_or(
        torch.abs(tensor1) > threshold1, torch.abs(tensor2) > threshold2
    )
    return logical_mask


def consistency(explanations: torch.Tensor) -> torch.Tensor:
    """
    Metric representing how similar are explanations of one photo.

    Metric representing how much do different explanations for the same model or same explanation for different models diverge.
    Maximal value of 1 represents identical explanations and values close to 0 represent greatly differing explanations.

    Parameters
    ----------
    explanations: torch.Tensor


    Returns
    -------
    float
        Value of the consistency metric for the input explanations.

    See Also
    --------
    replacetext : function description.

    Examples
    --------
    >>> function(x, y)
    answer
    """
    """
    Opis: Mierzy jak bardzo wyjaśnienia różnych modeli
    uczenia maszynowego są do siebie podobne.
    Argumenty wejściowe: Lista<torch.Tensor>
    (lista wyjaśnień które chcemy ze sobą porównać)
    Wartości wyjściowe: torch.Tensor (wynik metryki)
    :return:
    C(phi,...) =
    [max_{a,b}(||phi_{j}^{e->m_a} - phi_{j}^{e->m_b}||_2) + 1]^{-1},
    phi_j-wyjasnienie j tego zdjęcia lub
    [max_{a,b}(||phi_{j}^{e_a->m} - phi_{j}^{e_b->m}||_2) + 1]^{-1}
    """
    explanations_list = tensor_to_list_tensors(explanations, depth=2)
    diffs = [
        _matrix_norm_2(exp1, exp2)
        for exp1, exp2 in itertools.combinations(explanations_list, 2)
    ]
    return (1 / (max(diffs) + 1)).item()


def stability(explanator: Callable, image: torch.Tensor,
    images_to_compare: torch.Tensor, epsilon: float = 0.1,
) -> torch.Tensor:
    """
    Short description

    Long description

    Parameters
    ----------
    parameter1: parameter_type
        parameter_description

    Returns
    -------
    return_object: return_type
        return_description

    See Also
    --------
    replacetext : function description.

    Examples
    --------
    >>> function(x, y)
    answer
    """
    """
    Opis: Mierzy jak podobne wyjaśnienia otrzymamy
    dla podobnych danych wejściowych.
    Argumenty wejściowe:
    obiekt ‘callable’ (metoda zwracająca wyjaśnienie),
    torch.tensor (obrazek który chcemy wyjaśnić)
    Wartości wyjściowe: torch.Tensor (wynik metryki)
    :return:
    L(phi, X) = max_{x_j} (||x_i-x_j||_{2}/(||phi_i^{e->m} - phi_j^{e->m}||_{2}+1))
    https://github.com/sbobek/inxai/blob/main/inxai/global_metrics.py
    """
    images_list = tensor_to_list_tensors(images_to_compare, depth=1)
    # matrix_norm_2 over all 3 dimensions
    close_images = [
        other_image
        for other_image in images_list
        if _matrix_norm_2(image, other_image, sum_dim=-1).item() < epsilon
    ]
    close_images_tensor = torch.Tensor(close_images)
    close_images_explanations = explanator(close_images_tensor)
    image_explanation = explanator(image.unsqueeze(dim=0)).squeeze()
    image_dists = _matrix_norm_2(close_images_tensor, image, sum_dim=-1)
    expl_dists = _matrix_norm_2(close_images_explanations, image_explanation)
    return torch.max(image_dists / (expl_dists + 1)).item()


def _impact_ratio_helper(
    images_tensor: torch.Tensor,
    predictor: Callable[..., torch.Tensor],
    explanations: torch.Tensor,
    explanation_threshold: float,
    baseline: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Short description

    Long description

    Parameters
    ----------
    parameter1: parameter_type
        parameter_description

    Returns
    -------
    return_object: return_type
        return_description

    See Also
    --------
    replacetext : function description.

    Examples
    --------
    >>> function(x, y)
    answer
    """
    probabilities_original = predictor(images_tensor)
    # one explanation per image
    explanations_flat_bool = explanations[:, 0, :, :] > explanation_threshold
    modified_images = replace_masks(images_tensor, explanations_flat_bool, baseline)
    probabilities_modified = predictor(modified_images)
    return probabilities_original, probabilities_modified


def decision_impact_ratio(
    image_tensors: torch.Tensor,
    predictor: Callable[..., torch.Tensor],
    explanations: torch.Tensor,
    explanation_threshold: float,
    baseline: int,
) -> torch.Tensor:
    """
    Short description

    Long description

    Parameters
    ----------
    parameter1: parameter_type
        parameter_description

    Returns
    -------
    return_object: return_type
        return_description

    Opis: Jest to odsetek obserwacji, dla których po usunięciu
    obszaru wrażliwości (wskazanego przez wyjaśnienie)
    klasyfikacja modelu zmieniła się.
    Argumenty wejściowe: dataset, forward z modelu,
    funkcja wyjasnienia, baseline do podmiany pixeli
    Wartości wyjściowe: torch.Tensor (wynik metryki)
    :return:
    DIR = Suma po i (1 jeżeli D(x_i)=/=D(x_i-c_i) else 0)/N,
    D to klasyfikacja, c_i obszar krytyczny

    See Also
    --------
    replacetext : function description.

    Examples
    --------
    >>> function(x, y)
    answer
    """
    n = image_tensors.shape[0]
    # predictor returns probabilities in a tensor format
    probs_original, probs_modified = _impact_ratio_helper(
        image_tensors, predictor, explanations, explanation_threshold, baseline
    )
    _, preds_original = torch.max(probs_original, 1)
    _, preds_modified = torch.max(probs_modified, 1)
    value = torch.sum((preds_original != preds_modified).float()) / n
    return value.item()


def confidence_impact_ratio(
    images_tensor: torch.Tensor,
    predictor: Callable[..., torch.Tensor],
    explanations: torch.Tensor,
    explanation_threshold: float,
    baseline: int = 0,
) -> torch.Tensor:
    """
    Short description

    Long description

    Parameters
    ----------
    parameter1: parameter_type
        parameter_description

    Returns
    -------
    return_object: return_type
        return_description

    See Also
    --------
    replacetext : function description.

    Examples
    --------
    >>> function(x, y)
    answer
    """
    """
    Opis: Średni spadek estymowanego prawdopodobieństwa
    klasyfikacji po zasłonięciu obszaru wrażliwości.
    Argumenty wejściowe: dataset, funkcja na probsy z modelu,
    funkcja wyjasnienia, baseline do podmiany pixeli
    Wartości wyjściowe: torch.Tensor (wynik metryki)
    :return:
    CIR = Suma po i max(C(x_i)-C(x_i-c_i), 0)/N , C to probabilities, c_i obszar krytyczny
    """
    probs_original, probs_modified = _impact_ratio_helper(
        images_tensor, predictor, explanations, explanation_threshold, baseline
    )
    probs_max_original, _ = torch.max(probs_original, 1)
    probs_max_modified, _ = torch.max(probs_modified, 1)
    value = torch.sum(probs_max_original - probs_max_modified) / images_tensor.shape[0]
    return value.item()


def accordance_recall(
    explanations: torch.Tensor, masks: torch.Tensor, threshold: float = 0.0
) -> torch.Tensor:
    """
    Short description

    Long description

    Parameters
    ----------
    parameter1: parameter_type
        parameter_description

    Returns
    -------
    return_object: return_type
        return_description

    See Also
    --------
    replacetext : function description.

    Examples
    --------
    >>> function(x, y)
    answer
    """
    """
        Opis: Mierzy jaką część maski wykryło wyjaśnienie.
        Argumenty wejściowe: torch.Tensor (maska obrazu),
        torch.Tensor (wyjaśnienie/obszar krytyczny),
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
    # dla jednego wyjasnienia, wiec
    # wymiar explanations.shape = (n, 1, width, height) mask = (n, width, height)
    squeezed_expl = explanations.squeeze(dim=1)
    overlaping_area = _intersection_mask(squeezed_expl, masks, threshold1=threshold)
    divisor = torch.sum(torch.abs(masks) != 0, dim=(-2, -1))
    value = torch.sum(overlaping_area, dim=(-2, -1)) / divisor
    return value


def accordance_precision(
    explanations: torch.Tensor, masks: torch.Tensor, threshold: float = 0.0
) -> torch.Tensor:
    """
    Short description

    Long description

    Parameters
    ----------
    parameter1: parameter_type
        parameter_description

    Returns
    -------
    return_object: return_type
        return_description

    See Also
    --------
    replacetext : function description.

    Examples
    --------
    >>> function(x, y)
    answer
    """
    """
        Opis: mierzy jaką część wyjaśnienia stanowiła maska.
        Argumenty wejściowe: torch.Tensor (maska obrazu),
        torch.Tensor (wyjasnienie/obszar krytyczny),
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
    squeezed_expl = explanations.squeeze(dim=1)
    overlaping_area = _intersection_mask(squeezed_expl, masks, threshold1=threshold)
    divisor = torch.sum(torch.abs(squeezed_expl) > threshold, dim=(-2, -1))
    value = torch.sum(overlaping_area, dim=(-2, -1)) / divisor
    return value


def F1_score(
    explanations: torch.Tensor, masks: torch.Tensor, threshold: float = 0.0
) -> float:
    """
    Short description

    Long description

    Parameters
    ----------
    parameter1: parameter_type
        parameter_description

    Returns
    -------
    return_object: return_type
        return_description


    See Also
    --------
    replacetext : function description.

    Examples
    --------
    >>> function(x, y)
    answer
    """
    """
        Opis: Średnia harmoniczna Accordance recall i Accordance precision.
        Argumenty wejściowe: torch.Tensor (maska obrazu),
        torch.Tensor (wyjasnienie/obszar krytyczny),
        torch.Tensor/skalar (opcjonalnie,
        jaka minimalna wartość w wyjaśnieniu ma być uznana za ważną)
        Wartości wyjściowe: torch.Tensor (wynik metryki)
        :return:
        Let S(𝑥) be the suspicious pneumonia area that is annotated
    by the clinician for image x and let F(𝑥) be the critical area that
    is identified by the interpretation method
        Accordance F1 = 1/N * sum_i (2*(recall_i + precision_i)/(recall_i*precision_i))
    """
    acc_recall = accordance_recall(explanations, masks, threshold=threshold)
    acc_prec = accordance_precision(explanations, masks, threshold=threshold)
    values = 2 * (acc_recall * acc_prec) / (acc_recall + acc_prec)
    value = torch.sum(values) / values.shape[0]
    return value.item()


def intersection_over_union(
    explanations: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5
) -> float:
    """
    Short description

    Long description

    Parameters
    ----------
    parameter1: parameter_type
        parameter_description

    Returns
    -------
    return_object: return_type
        return_description

    See Also
    --------
    replacetext : function description.

    Examples
    --------
    >>> function(x, y)
    answer
    """
    """
        Opis: Pole iloczynu maski i wyjaśnienia podzielone przez
        pole sumy maski i wyjaśnienia.
        Argumenty wejściowe: torch.Tensor (maska obrazu),
        torch.Tensor (wyjasnienie/obszar krytyczny),
        torch.Tensor/skalar (opcjonalnie,
        jaka minimalna wartość w wyjaśnieniu ma być uznana za ważną)
        Wartości wyjściowe: torch.Tensor (wynik metryki)
        :return:
        Let S(𝑥) be the suspicious pneumonia area that is annotated
    by the clinician for image x and let F(𝑥) be the critical area that
    is identified by the interpretation method
        IOU=1/N * sum_i(S(x_i) cz. wspolna F(x_i)/S(x_i) suma F(x_i))
    """
    squeezed_expl = explanations.squeeze(dim=1)
    intersections = _intersection_mask(squeezed_expl, masks, threshold1=threshold)
    union_masks = _union_mask(squeezed_expl, masks, threshold1=threshold)
    values = torch.sum(intersections, dim=(-2, -1)) / torch.sum(
        union_masks, dim=(-2, -1)
    )
    value = torch.sum(values) / values.shape[0]
    return value.item()


def ensemble_score(
    weights: Union[List, torch.Tensor],
    metrics_scores: Union[List[torch.Tensor], torch.Tensor, List[float]],
) -> torch.Tensor:
    """
    Short description

    Long description

    Parameters
    ----------
    parameter1: parameter_type
        parameter_description

    Returns
    -------
    return_object: return_type
        return_description

    See Also
    --------
    replacetext : function description.

    Examples
    --------
    >>> function(x, y)
    answer
    """
    """
    Opis: średnia ważona innych metryk. Ensemble_score(wagi, metryki) -> torch.tensor
    Argumenty wejściowe:
    torch.Tensor/Lista (lista z wagami dla poszczególnych metryk),
    torch.Tensor/Lista (lista z wynikami poszczególnych metryk)
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
