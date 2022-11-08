
def Consistency():
    pass
#Opis: Mierzy jak bardzo wyjaśnienia różnych modeli uczenia maszynowego są do siebie podobne.
#Argumenty wejściowe: Lista<Tensor> (lista wyjaśnień które chcemy ze sobą porównać)
#Wartości wyjściowe: Tensor (wynik metryki)

def Stability():
    pass
#Opis: Mierzy jak podobne wyjaśnienia otrzymamy dla podobnych danych wejściowych.
#Argumenty wejściowe: obiekt ‘callable’ (metoda zwracająca wyjaśnienie), tensor (obrazek który chcemy wyjaśnić)
#Wartości wyjściowe: Tensor (wynik metryki)

def Decision_impact_ratio():
    pass
#Opis: Jest to odsetek obserwacji, dla których po usunięciu obszaru wrażliwości (wskazanego przez wyjaśnienie) klasyfikacja modelu zmieniła się.
#Argumenty wejściowe: dataset, forward z modelu, funkcja wyjasnienia, baseline do podmiany pixeli
#Wartości wyjściowe: Tensor (wynik metryki)

def Confidence_impact_ratio():
    pass
#Opis: Średni spadek estymowanego prawdopodobieństwa klasyfikacji po zasłonięciu obszaru wrażliwości.
#Argumenty wejściowe: dataset, funkcja na probsy z modelu, funkcja wyjasnienia, baseline do podmiany pixeli
#Wartości wyjściowe: Tensor (wynik metryki)

def Accordance_recall():
    pass
#Opis: Mierzy jaką część maski wykryło wyjaśnienie.
#Argumenty wejściowe: Tensor (maska obrazu), Tensor (wyjaśnienie/obszar krytyczny), Tensor/skalar (opcjonalnie, jaka minimalna wartość w wyjaśnieniu ma być uznana za ważną)
#Wartości wyjściowe: Tensor (wynik metryki)

def Accordance_precision():
    pass
#Opis: mierzy jaką część wyjaśnienia stanowiła maska.
#Argumenty wejściowe: Tensor (maska obrazu), Tensor (wyjasnienie/obszar krytyczny), Tensor/skalar (opcjonalnie, jaka minimalna wartość w wyjaśnieniu ma być uznana za ważną)
#Wartości wyjściowe: Tensor (wynik metryki)

def F1_score():
    pass
#Opis: Średnia harmoniczna Accordance recall i Accordance precision.
#Argumenty wejściowe: Tensor (maska obrazu), Tensor (wyjasnienie/obszar krytyczny), Tensor/skalar (opcjonalnie, jaka minimalna wartość w wyjaśnieniu ma być uznana za ważną)
#Wartości wyjściowe: Tensor (wynik metryki)

def Intersection_over_Union():
    pass
#Opis: Pole iloczynu maski i wyjaśnienia podzielone przez pole sumy maski i wyjaśnienia.
#Argumenty wejściowe: Tensor (maska obrazu), Tensor (wyjasnienie/obszar krytyczny), Tensor/skalar (opcjonalnie, jaka minimalna wartość w wyjaśnieniu ma być uznana za ważną)
#Wartości wyjściowe: Tensor (wynik metryki)

def Ensemble_score():
    pass
#Opis: średnia ważona innych metryk. Ensemble_score(wagi, metryki) -> tensor
#Argumenty wejściowe: Tensor/Lista (lista z wagami dla poszczególnych metryk), Tensor/Lista (lista z wynikami poszczególnych metryk)
#Wartości wyjściowe: Tensor (wynik metryki)
