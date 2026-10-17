"""Entrena YOLOv8 con las tres clases y evalúa en la división test de VOC 2007.

    python scripts/entrenar.py --modelo yolov8s.pt --epocas 100
    python scripts/entrenar.py --evaluar models/yolov8s_voc3.pt   # solo la evaluación en test

Si `mlflow` está instalado, Ultralytics registra la corrida en ./mlruns.
Al final copia los mejores pesos a models/ y escribe sus métricas en JSON.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO  # noqa: E402

from deteccion.metricas import por_clase, resumen  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--datos", type=Path, default=Path("data/data.yaml"))
p.add_argument("--modelo", default="yolov8s.pt")
p.add_argument("--epocas", type=int, default=100)
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--lote", type=int, default=16)
p.add_argument("--nombre", default="yolov8s_voc3")
p.add_argument("--evaluar", type=Path, help="solo evaluar estos pesos en test")
args = p.parse_args()

if args.evaluar:
    destino = args.evaluar
    epocas = None
else:
    modelo = YOLO(args.modelo)
    modelo.train(
        data=str(args.datos), epochs=args.epocas, imgsz=args.imgsz, batch=args.lote, patience=20,
        seed=42, deterministic=True, name=args.nombre, exist_ok=True,
    )
    # La carpeta real la decide Ultralytics (con un project relativo la anida
    # en runs/detect/...); por eso el cuaderno 05 no encontraba best.pt.
    mejor = Path(modelo.trainer.save_dir) / "weights" / "best.pt"
    destino = Path("models") / f"{args.nombre}.pt"
    destino.parent.mkdir(exist_ok=True)
    shutil.copy(mejor, destino)
    epocas = modelo.trainer.epoch + 1

entrenado = YOLO(destino)
metricas = entrenado.val(data=str(args.datos), split="test", imgsz=args.imgsz, name=f"{destino.stem}_test", exist_ok=True)
informe = {
    "pesos": destino.name, "epocas_entrenadas": epocas, "imgsz": args.imgsz,
    "test": resumen(metricas), "por_clase": por_clase(metricas, entrenado.names),
}
destino.with_suffix(".json").write_text(json.dumps(informe, indent=2) + "\n")
print(json.dumps(informe, indent=2))
