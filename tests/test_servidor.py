import io
from pathlib import Path

import pytest
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
MODELO = RAIZ / "models" / "yolov8s_voc3.pt"
FOTO = Path(__file__).parent / "fixtures" / "paseo_turku.jpg"

pytestmark = pytest.mark.skipif(not MODELO.exists(), reason="falta el modelo entrenado")


@pytest.fixture(scope="module")
def cliente():
    from app.servidor import create_app
    return create_app(MODELO).test_client()


def enviar(cliente, datos: bytes, **form):
    return cliente.post("/predict", data={"image": (io.BytesIO(datos), "foto.jpg"), **form},
                        content_type="multipart/form-data")


def test_salud_e_informacion(cliente):
    assert cliente.get("/health").get_json() == {"estado": "ok", "modelo": "yolov8s_voc3.pt"}
    assert cliente.get("/model-info").get_json()["clases"] == ["person", "car", "dog"]
    assert b"Detector de personas" in cliente.get("/").data


def test_detecta_persona_perro_y_auto_en_una_foto_real(cliente):
    r = enviar(cliente, FOTO.read_bytes())
    assert r.status_code == 200
    datos = r.get_json()
    assert datos["conteo"]["person"] >= 1
    assert datos["conteo"]["dog"] >= 1
    assert datos["conteo"]["car"] >= 1
    assert datos["imagen_anotada"].startswith("data:image/jpeg;base64,")
    confianzas = [d["confianza"] for d in datos["detecciones"]]
    assert confianzas == sorted(confianzas, reverse=True)
    assert all(c >= 0.25 for c in confianzas)


def test_una_confianza_mas_alta_deja_menos_detecciones(cliente):
    bajas = enviar(cliente, FOTO.read_bytes(), conf="0.1").get_json()["detecciones"]
    altas = enviar(cliente, FOTO.read_bytes(), conf="0.8").get_json()["detecciones"]
    assert len(altas) < len(bajas)


def test_acepta_png_con_transparencia(cliente):
    buffer = io.BytesIO()
    Image.new("RGBA", (64, 64), (255, 0, 0, 128)).save(buffer, format="PNG")
    r = enviar(cliente, buffer.getvalue())
    assert r.status_code == 200
    assert r.get_json()["detecciones"] == []


@pytest.mark.parametrize("datos, form, error", [
    (b"no es una imagen", {}, "no es una imagen"),
    (None, {}, "Falta la imagen"),
])
def test_errores(cliente, datos, form, error):
    if datos is None:
        r = cliente.post("/predict", data={}, content_type="multipart/form-data")
    else:
        r = enviar(cliente, datos, **form)
    assert r.status_code == 400
    assert error in r.get_json()["error"]


def test_confianza_fuera_de_rango(cliente):
    r = enviar(cliente, FOTO.read_bytes(), conf="2")
    assert r.status_code == 400
