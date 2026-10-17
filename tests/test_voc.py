from pathlib import Path

from PIL import Image

from deteccion.voc import convertir_anotacion, preparar


def xml(objetos, ancho=500, alto=400):
    cuerpo = "".join(
        f"<object><name>{n}</name><difficult>{d}</difficult><bndbox><xmin>{a}</xmin><ymin>{b}</ymin>"
        f"<xmax>{c}</xmax><ymax>{e}</ymax></bndbox></object>"
        for n, d, (a, b, c, e) in objetos
    )
    return f"<annotation><size><width>{ancho}</width><height>{alto}</height><depth>3</depth></size>{cuerpo}</annotation>"


def test_convierte_a_yolo_normalizado():
    lineas = convertir_anotacion(xml([("dog", 0, (101, 51, 300, 250))]))
    # VOC cuenta los píxeles desde 1: la caja va de (100, 50) a (300, 250).
    assert lineas == ["2 0.400000 0.375000 0.400000 0.500000"]


def test_ignora_otras_clases_y_objetos_dificiles():
    lineas = convertir_anotacion(xml([
        ("cat", 0, (1, 1, 10, 10)),
        ("person", 1, (1, 1, 10, 10)),
        ("car", 0, (1, 1, 500, 400)),
    ]))
    assert lineas == ["1 0.500000 0.500000 1.000000 1.000000"]


def test_recorta_cajas_que_se_salen_de_la_imagen():
    clase, xc, yc, w, h = convertir_anotacion(xml([("person", 0, (0, 0, 600, 450))]))[0].split()
    assert (float(xc), float(yc), float(w), float(h)) == (0.5, 0.5, 1.0, 1.0)


def _voc_minimo(raiz: Path):
    """VOC con 4 imágenes de trainval (una sin clases útiles) y 2 de test."""
    for carpeta in ("Annotations", "JPEGImages", "ImageSets/Main"):
        (raiz / carpeta).mkdir(parents=True)
    anotaciones = {
        "a": [("person", 0, (10, 10, 100, 100))],
        "b": [("dog", 0, (10, 10, 100, 100)), ("car", 0, (20, 20, 50, 50))],
        "c": [("cat", 0, (10, 10, 100, 100))],
        "d": [("car", 0, (5, 5, 60, 60))],
        "t1": [("person", 0, (10, 10, 50, 50))],
        "t2": [("dog", 1, (10, 10, 50, 50))],
    }
    for id_, objetos in anotaciones.items():
        (raiz / "Annotations" / f"{id_}.xml").write_text(xml(objetos))
        Image.new("RGB", (500, 400)).save(raiz / "JPEGImages" / f"{id_}.jpg")
    (raiz / "ImageSets/Main/trainval.txt").write_text("a\nb\nc\nd\n")
    (raiz / "ImageSets/Main/test.txt").write_text("t1\nt2\n")


def test_prepara_la_estructura_de_yolo_sin_mezclar_test(tmp_path):
    voc = tmp_path / "VOC2007"
    _voc_minimo(voc)
    cuentas = preparar(voc, voc, tmp_path / "salida", fraccion_val=0.25)
    # c no tiene ninguna de las tres clases; t2 solo tiene un perro "difficult".
    assert cuentas["train"] + cuentas["val"] == 3
    assert cuentas["test"] == 1
    salida = tmp_path / "salida"
    assert sorted(p.stem for p in (salida / "images" / "test").iterdir()) == ["t1"]
    entrenamiento = {p.stem for d in ("train", "val") for p in (salida / "labels" / d).iterdir()}
    assert entrenamiento == {"a", "b", "d"}
    yaml = (salida / "data.yaml").read_text()
    assert "0: person" in yaml and "2: dog" in yaml and "test: images/test" in yaml


def test_la_division_es_reproducible(tmp_path):
    voc = tmp_path / "VOC2007"
    _voc_minimo(voc)
    preparar(voc, voc, tmp_path / "s1", fraccion_val=0.25)
    preparar(voc, voc, tmp_path / "s2", fraccion_val=0.25)
    assert [p.name for p in (tmp_path / "s1/images/val").iterdir()] == [p.name for p in (tmp_path / "s2/images/val").iterdir()]
