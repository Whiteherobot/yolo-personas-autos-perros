"""Convierte Pascal VOC 2007 al formato de YOLO, solo con las tres clases.

Se usa la división oficial: `trainval` para entrenar (un 10 % queda para
validación) y `test` para evaluar, así ninguna imagen de prueba se ve al
entrenar. Los objetos marcados como `difficult` se omiten, como en la
evaluación oficial de VOC.
"""
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from . import CLASES


def convertir_anotacion(xml: Path | str, clases: list[str] = CLASES) -> list[str]:
    """Líneas YOLO ("clase xc yc ancho alto", normalizadas) de un XML de VOC (ruta o texto)."""
    raiz = ET.parse(xml).getroot() if isinstance(xml, Path) else ET.fromstring(xml)
    tamano = raiz.find("size")
    ancho, alto = float(tamano.find("width").text), float(tamano.find("height").text)
    lineas = []
    for obj in raiz.iter("object"):
        nombre = obj.find("name").text.strip()
        dificil = obj.find("difficult")
        if nombre not in clases or (dificil is not None and dificil.text.strip() == "1"):
            continue
        caja = obj.find("bndbox")
        x1, y1, x2, y2 = (float(caja.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        # VOC numera los píxeles desde 1.
        x1, y1 = max(x1 - 1, 0), max(y1 - 1, 0)
        x2, y2 = min(x2, ancho), min(y2, alto)
        lineas.append(
            f"{clases.index(nombre)} {(x1 + x2) / 2 / ancho:.6f} {(y1 + y2) / 2 / alto:.6f} "
            f"{(x2 - x1) / ancho:.6f} {(y2 - y1) / alto:.6f}"
        )
    return lineas


def _ids(voc: Path, division: str) -> list[str]:
    return (voc / "ImageSets" / "Main" / f"{division}.txt").read_text().split()


def preparar(voc_trainval: Path, voc_test: Path, salida: Path, fraccion_val: float = 0.1, semilla: int = 42) -> dict:
    """Arma salida/{images,labels}/{train,val,test} y salida/data.yaml. Devuelve cuántas imágenes hay por división."""
    trainval = _ids(voc_trainval, "trainval")
    random.Random(semilla).shuffle(trainval)
    corte = int(len(trainval) * fraccion_val)
    divisiones = {
        "train": (voc_trainval, sorted(trainval[corte:])),
        "val": (voc_trainval, sorted(trainval[:corte])),
        "test": (voc_test, _ids(voc_test, "test")),
    }
    cuentas = {}
    for division, (voc, ids) in divisiones.items():
        (salida / "images" / division).mkdir(parents=True, exist_ok=True)
        (salida / "labels" / division).mkdir(parents=True, exist_ok=True)
        n = 0
        for id_ in ids:
            lineas = convertir_anotacion(voc / "Annotations" / f"{id_}.xml")
            if not lineas:
                continue  # sin ninguna de las tres clases
            shutil.copy2(voc / "JPEGImages" / f"{id_}.jpg", salida / "images" / division / f"{id_}.jpg")
            (salida / "labels" / division / f"{id_}.txt").write_text("\n".join(lineas) + "\n")
            n += 1
        cuentas[division] = n
    (salida / "data.yaml").write_text(
        f"path: {salida.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASES))
    )
    return cuentas
