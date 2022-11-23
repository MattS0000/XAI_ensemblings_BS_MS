from typing import Callable, Union, List, Tuple
import itertools
import torch

function = lambda x, y: (x, y)
x=1
y=2

def replace_masks(
        images: torch.Tensor, replacement_index: torch.Tensor, value: Union[int, float] = 0
) -> torch.Tensor:
    """
    Replaces values in images where masks exist.

    Replaces data in the images Tensor with one value in the spots where masks Tensor is True.

    Parameters
    ----------
    images: torch.Tensor
        Tensor of any shape, in most cases 4D Tensor of the images with shape (number of photos, RGB channel, height, width)
    replacement_index: torch.Tensor
        Boolean Tensor of shape same as images or in case of the 4D images Tensor
        3D Tensor where true corresponds index to be replaced with shape (number of photos, height, width)
    value: int or float
        Value to use for replacing the data with.

    Returns
    -------
    torch.Tensor
        Tensor of same shape as input with the replaced data.

    See Also
    --------
    _impact_ratio_helper : function description.
    decision_impact_ratio : function description.
    confidence_impact_ratio : function description.

    Examples
    --------
    >>> function(x, y)
    answer
    """
    temp_images = torch.clone(images)
    # 3D Tensor needs to be reshaped over RGB channel for the indexing of images
    if len(replacement_index.shape) == 3 and len(temp_images.shape) == 4:
        replacement_index = replacement_index.unsqueeze(dim=1).repeat(1, temp_images.shape[1], 1, 1)
    temp_images[replacement_index] = value
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
    consistency : function description.
    stability : function description.

    Examples
    --------
    >>> function(x, y)
    answer
    """
    # squeezing couses returned tensors to have reduced dimensions
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


def matrix_2_norm(
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
    consistency : function description.
    stability :

    Examples
    --------
    >>> function(x, y)
    answer
    """
    if sum_dim is not None and sum_dim < 0:
        sum_dim = sum_dim+2
    difference = (matrix1 - matrix2).float()
    norm = torch.linalg.matrix_norm(difference, ord=2)
    # manual extension of the norm calculation to the sum_dim dimension
    if sum_dim is not None:
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
    accordance_recall : function description.
    accordance_precision :
    intersection_over_union :
    _union_mask :

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
    intersection_over_union : function description.
    _intersection_mask :

    Examples
    --------
    >>> function(x, y)
    answer
    """
    logical_mask = torch.logical_or(
        torch.abs(tensor1) > threshold1, torch.abs(tensor2) > threshold2
    )
    return logical_mask


def consistency(explanations: torch.Tensor) -> float:
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
    stability : function description.
    tensor_to_list_tensors :
    matrix_2_norm :

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
    explanations_list = tensor_to_list_tensors(explanations, depth=1)
    diffs = [
        matrix_2_norm(exp1, exp2, sum_dim=0)
        for exp1, exp2 in itertools.combinations(explanations_list, 2)
    ]
    return (1 / (max(diffs) + 1)).item()


def stability(explanator: Callable, image: torch.Tensor,
              images_to_compare: torch.Tensor, epsilon: float = 500.0,
              ) -> torch.Tensor:
    """
    Short description

    Long description

    Parameters
    ----------
    explanator: Callable
        parameter_description
    image: torch.Tensor
        parameter_description
    images_to_compare: torch.Tensor
        parameter_description
    epsilon: float
        parameter_description

    Returns
    -------
    torch.Tensor
        return_description

    See Also
    --------
    consistency : function description.
    tensor_to_list_tensors :
    matrix_2_norm :

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
    # matrix 2-norm over all 3 dimensions
    close_images = [
        other_image
        for other_image in images_list
        if matrix_2_norm(image, other_image, sum_dim=0).item() < epsilon
    ]
    close_images_tensor = torch.stack(close_images)
    close_images_explanations = explanator(close_images_tensor)
    image_explanation = explanator(image.unsqueeze(dim=0)).squeeze(dim=0)
    # matrix_2_norm works if one tensor is of one shape bigger, casts the other to the correct size
    image_dists = matrix_2_norm(close_images_tensor, image, sum_dim=1)
    expl_dists = matrix_2_norm(close_images_explanations, image_explanation, sum_dim=1)
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
    images_tensor: torch.Tensor
        parameter_description
    predictor: Callable[..., torch.Tensor]
        parameter_description
    explanations: torch.Tensor
        parameter_description
    explanation_threshold: float
        parameter_description
    baseline: int
        parameter_description

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
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
    explanations_boolean = explanations > explanation_threshold
    modified_images = replace_masks(images_tensor, explanations_boolean, baseline)
    probabilities_modified = predictor(modified_images)
    return probabilities_original, probabilities_modified


def decision_impact_ratio(
        images_tensors: torch.Tensor,
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
    image_tensors: torch.Tensor
        parameter_description
    predictor: Callable[..., torch.Tensor]
        parameter_description
    explanations: torch.Tensor
        parameter_description
    explanation_threshold: float
        parameter_description
    baseline: int
        parameter_description

    Returns
    -------
    torch.Tensor
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
    n = images_tensors.shape[0]
    # predictor returns probabilities in a tensor format
    probs_original, probs_modified = _impact_ratio_helper(
        images_tensors, predictor, explanations, explanation_threshold, baseline
    )
    _, preds_original = torch.max(probs_original, 1)
    _, preds_modified = torch.max(probs_modified, 1)
    value = torch.sum((preds_original != preds_modified).float()) / n
    return value.item()


def confidence_impact_ratio(
        images_tensors: torch.Tensor,
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
    images_tensors: torch.Tensor
        parameter_description
    predictor: Callable[..., torch.Tensor]
        parameter_description
    explanations: torch.Tensor
        parameter_description
    explanation_threshold: float
        parameter_description
    baseline: int
        parameter_description

    Returns
    -------
    torch.Tensor
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
        images_tensors, predictor, explanations, explanation_threshold, baseline
    )
    probs_max_original, _ = torch.max(probs_original, 1)
    probs_max_modified, _ = torch.max(probs_modified, 1)
    value = torch.sum(probs_max_original - probs_max_modified) / images_tensors.shape[0]
    return value.item()


def accordance_recall(
        explanations: torch.Tensor, masks: torch.Tensor, threshold: float = 0.0
) -> torch.Tensor:
    """
    Short description

    Long description

    Parameters
    ----------
    explanations: torch.Tensor
        parameter_description
    masks: torch.Tensor
        parameter_description
    threshold: float
        parameter_description

    Returns
    -------
    torch.Tensor
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
    # logical mask, one explanation per image
    # explanations.shape = (n, x, width, height), mask.shape = (n, width, height)
    # squeeze explanation to be of same shape as masks
    reshaped_mask = masks.unsqueeze(dim=1).repeat(1, explanations.shape[1], 1, 1)
    overlapping_area = _intersection_mask(explanations, reshaped_mask, threshold1=threshold)
    divisor = torch.sum(reshaped_mask != 0, dim=(-3, -2, -1))
    value = torch.sum(overlapping_area, dim=(-3, -2, -1)) / divisor
    return value


def accordance_precision(
        explanations: torch.Tensor, masks: torch.Tensor, threshold: float = 0.0
) -> torch.Tensor:
    """
    Short description

    Long description

    Parameters
    ----------
    explanations: torch.Tensor
        parameter_description
    masks: torch.Tensor
        parameter_description
    threshold: float
        parameter_description

    Returns
    -------
    torch.Tensor
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
    reshaped_mask = masks.unsqueeze(dim=1).repeat(1, explanations.shape[1], 1, 1)
    overlapping_area = _intersection_mask(explanations, reshaped_mask, threshold1=threshold)
    divisor = torch.sum(torch.abs(explanations) > threshold, dim=(-3, -2, -1))
    value = torch.sum(overlapping_area, dim=(-3, -2, -1)) / divisor
    return value


def F1_score(
        explanations: torch.Tensor, masks: torch.Tensor, threshold: float = 0.0
) -> float:
    """
    Short description

    Long description

    Parameters
    ----------
    explanations: torch.Tensor
        parameter_description
    masks: torch.Tensor

    threshold: float

    Returns
    -------
    float
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
    explanations: torch.Tensor
        parameter_description
    masks: torch.Tensor

    threshold: float

    Returns
    -------
    float
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
    # one explanation per image
    reshaped_mask = masks.unsqueeze(dim=1).repeat(1, explanations.shape[1], 1, 1)
    intersections = _intersection_mask(explanations, reshaped_mask, threshold1=threshold)
    union_masks = _union_mask(explanations, reshaped_mask, threshold1=threshold)
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
    weights: Union[List, torch.Tensor]
        parameter_description
    metrics_scores: Union[List[torch.Tensor], torch.Tensor, List[float]]
        parameter_description

    Returns
    -------
    Torch.Tensor
        return_description

    See Also
    --------
    replacetext : function description.

    Examples
    --------
    >>> 1 * 3 + 2 * 5
    13
    >>> ensemble_score([1, 2], [3, 5])
    13
    >>> 1 * 5 + 2 * 3
    11
    >>> ensemble_score([1, 2], [5, 3])
    11
    """
    return sum(
        [
            weight * metric_score
            for (weight, metric_score) in zip(weights, metrics_scores)
        ]
    )
