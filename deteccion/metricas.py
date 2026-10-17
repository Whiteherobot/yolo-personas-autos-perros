"""Métricas por clase a partir del resultado de `YOLO.val()`."""


def por_clase(metricas, nombres: dict[int, str]) -> dict[str, dict[str, float]]:
    """AP50 y AP50-95 de cada clase.

    `box.ap50` es el AP50 por clase y `box.maps` el AP50-95 por clase (indexado
    por id de clase); confundirlos fue el error del cuaderno 05 original.
    """
    caja = metricas.box
    resultado = {}
    for i, indice in enumerate(caja.ap_class_index):
        resultado[nombres[int(indice)]] = {"ap50": float(caja.ap50[i]), "ap50_95": float(caja.ap[i])}
    return resultado


def resumen(metricas) -> dict[str, float]:
    caja = metricas.box
    return {"map50": float(caja.map50), "map50_95": float(caja.map), "precision": float(caja.mp), "recall": float(caja.mr)}
