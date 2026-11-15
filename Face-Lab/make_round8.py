"""Round 8 Colab bundle: colab/Face-Lab-code-r8.zip (code + cache/rav_ft_pack.npz) and colab/train_face_v2_round8_rav_ft_colab.ipynb.
  python make_round8.py"""
import json
import zipfile
from pathlib import Path

H = Path(__file__).resolve().parent

CELLS = [
    ("markdown", """# Face v2 — round 8: fine-tune HSEmotion B2 on RAVDESS 24 actors (leave-actors-out)
Same folds as the face models' comparison (actors[i::6]): 6 folds → out-of-fold outputs for every RAVDESS crop (evaluated on the laptop
with the same samples / metrics: current head 0.723, blendshape model 0.591), then fold "all" → `best.onnx` for the web.
Everything is written to `MyDrive/face-v2/out/rav_ft/` as it finishes, so if Colab disconnects just **Run all** again: finished folds are skipped.

Needs in `MyDrive/face-v2/`: `Face-Lab-code-r8.zip`. Runtime → **T4 GPU**. About 1–1.5 h."""),
    ("code", """import shutil, subprocess
if shutil.which('nvidia-smi'):
    print(subprocess.run(['nvidia-smi','-L'],capture_output=True,text=True).stdout)
else:
    raise SystemExit('NO GPU: Runtime > Change runtime type > T4 GPU, then Run all again')"""),
    ("code", """from google.colab import drive
drive.mount('/content/drive')
import os, zipfile, glob, hashlib
D='/content/drive/MyDrive/face-v2'; OUT=f'{D}/out/rav_ft'
os.makedirs(OUT, exist_ok=True); os.makedirs('/content/Main_demo', exist_ok=True)
Z = f'{D}/Face-Lab-code-r8.zip'
if not os.path.exists(Z):                      # uploaded in pieces (r8_parts/r8.zip.p00 …) → join them back
    ps = sorted(glob.glob(f'{D}/r8_parts/r8.zip.p*')); print(len(ps), 'parts')
    Z = '/content/r8.zip'
    with open(Z, 'wb') as o:
        for p in ps: o.write(open(p, 'rb').read())
h = hashlib.sha256(open(Z, 'rb').read()).hexdigest()
assert h == ZIP_SHA, ('zip checksum mismatch', h)
zipfile.ZipFile(Z).extractall('/content/Main_demo')
%cd /content/Main_demo/Face-Lab
!pip -q install onnx onnx2torch onnxruntime
!mkdir -p hse && wget -q -O hse/enet_b2_8.onnx https://github.com/av-savchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/onnx/enet_b2_8.onnx && ls -la hse cache
print(sorted(os.listdir(OUT)))"""),
    ("code", """!python ft_rav.py --out {OUT} --folds 0,1,2,3,4,5,all --epochs 2 --lr 3e-5 --kd 0.5"""),
    ("code", """# pack the outputs in parts < 9 MB (so they can be pulled back to the laptop) → out/rav_ft_parts/
import glob
P = f'{D}/out/rav_ft_parts'; os.makedirs(P, exist_ok=True)
for f in sorted(glob.glob(f'{OUT}/*')):
    b = open(f,'rb').read(); n = os.path.basename(f)
    for i in range(0, len(b), 9000000):
        open(f'{P}/{n}.part{i // 9000000}','wb').write(b[i:i + 9000000])
print(len(os.listdir(P)), 'parts:', sorted(os.listdir(P))[:5], '...')"""),
]


def main():
    import hashlib, sys
    zp = H / "colab" / "Face-Lab-code-r8.zip"
    if "--rebuild" in sys.argv or not zp.exists():      # the zip is kept as is otherwise (its checksum is what the uploaded parts match)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in ("data.py", "ft_rav.py"):
                z.write(H / f, f"Face-Lab/{f}")
            z.write(H / "cache" / "rav_ft_pack.npz", "Face-Lab/cache/rav_ft_pack.npz")
    sha = hashlib.sha256(zp.read_bytes()).hexdigest()
    cells = [(t, s.replace("ZIP_SHA", repr(sha))) for t, s in CELLS]
    nb = dict(nbformat=4, nbformat_minor=5, metadata={"accelerator": "GPU", "colab": {"gpuType": "T4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
              cells=[dict(cell_type=t, metadata={}, source=s, **({"outputs": [], "execution_count": None} if t == "code" else {})) for t, s in cells])
    (H / "colab" / "train_face_v2_round8_rav_ft_colab.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("ok", zp.stat().st_size / 1e6, "MB", sha)


if __name__ == "__main__":
    main()
