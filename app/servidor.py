"""API y página para detectar personas, autos y perros.

    flask --app app.servidor run          # o: python -m app.servidor

MODELO apunta a los pesos (por defecto models/yolov8s_voc3.pt).
"""
import base64
import io
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[1]
MODELO = Path(os.getenv("MODELO", RAIZ / "models" / "yolov8s_voc3.pt"))
MAX_BYTES = 10 * 1024 * 1024
CONF_DEFECTO = 0.25
IOU = 0.5


def _numero(valor, defecto, minimo, maximo):
    try:
        x = float(valor) if valor is not None else defecto
    except ValueError:
        return None
    return x if minimo <= x <= maximo else None


def create_app(ruta_modelo: Path = MODELO) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES
    # Se carga al crear la app (antes solo se cargaba al correr el archivo
    # directamente, y con `flask run` la API respondía que no había modelo).
    modelo = YOLO(str(ruta_modelo))
    nombres = modelo.names

    @app.get("/")
    def inicio():
        return render_template("index.html", clases=list(nombres.values()))

    @app.get("/health")
    def salud():
        return jsonify(estado="ok", modelo=ruta_modelo.name)

    @app.get("/model-info")
    def info():
        return jsonify(modelo=ruta_modelo.name, clases=list(nombres.values()), conf_defecto=CONF_DEFECTO, iou=IOU)

    @app.post("/predict")
    def predecir():
        archivo = request.files.get("image")
        if archivo is None or not archivo.filename:
            return jsonify(error="Falta la imagen (campo 'image')"), 400
        conf = _numero(request.form.get("conf"), CONF_DEFECTO, 0.01, 0.99)
        if conf is None:
            return jsonify(error="conf tiene que estar entre 0.01 y 0.99"), 400
        try:
            imagen = Image.open(archivo.stream)
            imagen = ImageOps.exif_transpose(imagen).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return jsonify(error="El archivo no es una imagen válida"), 400

        resultado = modelo.predict(imagen, conf=conf, iou=IOU, verbose=False)[0]
        detecciones = [
            {
                "clase": nombres[int(c)],
                "confianza": round(float(p), 3),
                "caja": [round(float(v), 1) for v in xyxy],
            }
            for c, p, xyxy in zip(resultado.boxes.cls, resultado.boxes.conf, resultado.boxes.xyxy)
        ]
        detecciones.sort(key=lambda d: -d["confianza"])
        conteo = {n: sum(d["clase"] == n for d in detecciones) for n in nombres.values()}

        anotada = Image.fromarray(resultado.plot()[..., ::-1])
        buffer = io.BytesIO()
        anotada.save(buffer, format="JPEG", quality=88)
        return jsonify(
            detecciones=detecciones,
            conteo=conteo,
            ancho=imagen.width,
            alto=imagen.height,
            conf=conf,
            imagen_anotada="data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(),
        )

    @app.errorhandler(413)
    def muy_grande(_):
        return jsonify(error="La imagen supera los 10 MB"), 413

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000)
