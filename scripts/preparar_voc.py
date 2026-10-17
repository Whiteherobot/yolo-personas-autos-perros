"""Descarga Pascal VOC 2007 (si hace falta) y lo deja en formato YOLO.

    python scripts/preparar_voc.py --voc ~/.cache/voc2007 --salida data
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deteccion.voc import preparar  # noqa: E402

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/{}"
ARCHIVOS = ["VOCtrainval_06-Nov-2007.zip", "VOCtest_06-Nov-2007.zip"]

p = argparse.ArgumentParser()
p.add_argument("--voc", type=Path, default=Path.home() / ".cache" / "voc2007")
p.add_argument("--salida", type=Path, default=Path("data"))
args = p.parse_args()

args.voc.mkdir(parents=True, exist_ok=True)
for archivo in ARCHIVOS:
    zip_ = args.voc / archivo
    if not zip_.exists():
        print(f"Descargando {archivo}...")
        subprocess.run(["curl", "-sSL", "-o", str(zip_), URL.format(archivo)], check=True)
        subprocess.run(["unzip", "-qo", str(zip_), "-d", str(args.voc)], check=True)

raiz = args.voc / "VOCdevkit" / "VOC2007"
# Los dos zip se descomprimen en la misma carpeta VOC2007.
cuentas = preparar(raiz, raiz, args.salida)
print("Imágenes por división:", cuentas)
