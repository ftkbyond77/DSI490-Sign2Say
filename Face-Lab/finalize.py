"""Turn the chosen training run into the deployable model folder and measure it.

  python finalize.py --run runs/<tag> --name v1
Writes models/face-model-<arch>-<name>/  {face_fer.onnx, face_fer.pt, config.json, model_card.json, README.md}  and  results/final_eval.json.
Test data is used here ONCE, after every choice (checkpoint, temperature, unknown thresholds) was made on validation data.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

import build_ferplus as fp
import data
import evaluate as ev
from models import FerNet, export_onnx

ROOT = Path(__file__).resolve().parent
MODELS = ROOT.parent / "models"
CROP = dict(scale=1.22, shift_up=0.16, points="MediaPipe 68 face points (modules/mediapipe_pose.MESH68)",
            note="measured on FER2013: the 68-point box covers 0.82 of a FER2013 image side (median, 129 faces) and sits 0.16 box-heights below its centre")


def choose_thresholds(yv, pv, quality_min=0.25, coverage_min=0.80):
    """min_conf from validation data: the smallest confidence at which accuracy of the answered frames is highest while at least
    `coverage_min` of frames are still answered. min_margin stays small (0.10): it only removes exact ties."""
    best = None
    for s in ev.selective(yv, pv, np.round(np.arange(0.30, 0.80, 0.01), 2)):
        if s["coverage"] >= coverage_min and (best is None or s["macro_f1"] > best["macro_f1"] + 1e-4):
            best = s
    return dict(min_conf=float(best["threshold"]), min_margin=0.10, min_quality=quality_min), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--out-dir", default=str(MODELS), help="where the model folder is written (default ../models; use candidates/ until it beats the shipped model)")
    ap.add_argument("--labels", default="fer2013", choices=["fer2013", "ferplus"], help="labels used to fit temperature + unknown thresholds on val (both are always reported on test)")
    ap.add_argument("--neutral-shift", type=float, default=0.6, help="deployment prior: logit bonus for neutral (signers are mostly neutral; free on FER2013)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # CPU works too (FP32; GPU runs use FP16 autocast)
    ck = torch.load(ROOT / a.run / "best.pt", map_location="cpu")
    net = FerNet(ck["arch"], ck["res"], pretrained=False)
    net.load_state_dict(ck["state"]); net.eval().to(dev)
    D = data.prepare(ck["args"]["seed"])
    (Xva, yva), (Xte, yte), clean = D["val"], D["test"], D["clean_test"]
    raw = data.load_raw()                                          # FER+ majority labels (10 annotators) for val and test; images without usable votes are left out
    V = fp.load({"train": raw["train"][2], "test": raw["test"][2]})
    tv, okv_all = fp.soft_targets(V["train"]); tt, okt = fp.soft_targets(V["test"])
    okv, yva_fp, yte_fp = okv_all[D["val_idx"]], tv[D["val_idx"]].argmax(1), tt.argmax(1)

    zv = ev.predict_logits(net, Xva, dev)
    zs, ys = (zv[okv], yva_fp[okv]) if a.labels == "ferplus" else (zv, yva)     # what calibration / thresholds are fitted on
    T = ev.fit_temperature(zs, ys)
    shift = {"neutral": a.neutral_shift} if a.neutral_shift else {}
    bias = np.array([shift.get(c, 0.0) for c in data.CLASSES])
    P = lambda z: ev.softmax(z + T * bias, T)                  # calibrated probabilities with the deployment prior (what the served model reports)   # noqa: E731
    pv = P(zv)
    unk, sel = choose_thresholds(ys, P(zs))

    zt = ev.predict_logits(net, Xte, dev)
    zt_f = ev.predict_logits(net, Xte, dev, flip=True)
    pt = P(zt)
    out = dict(run=a.run, arch=ck["arch"], res=ck["res"], temperature=T, thresholds=unk, thresholds_val_point=sel,
               val=ev.classification(yva, pv), val_calibration=ev.calibration(yva, pv),
               test=ev.classification(yte, pt), test_clean=ev.classification(yte[clean], pt[clean]),
               labels_for_fitting=a.labels, val_ferplus=ev.classification(yva_fp[okv], pv[okv]), test_ferplus=ev.classification(yte_fp[okt], pt[okt]),
               test_ferplus_clean=ev.classification(yte_fp[okt & clean], pt[okt & clean]), test_ferplus_calibration=ev.calibration(yte_fp[okt], pt[okt]),
               n_test_ferplus=int(okt.sum()), test_calibration=ev.calibration(yte, pt), test_calibration_raw=ev.calibration(yte, ev.softmax(zt)),
               test_flip_tta=ev.classification(yte, P(zt_f)), test_selective=ev.selective(yte, pt), n_test_clean=int(clean.sum()))
    # what the unknown rule does on test (frames below the thresholds are "unknown")
    conf, pred = pt.max(1), pt.argmax(1)
    srt = np.sort(pt, 1)
    ok = (conf >= unk["min_conf"]) & (srt[:, -1] - srt[:, -2] >= unk["min_margin"])
    out["test_with_unknown"] = dict(coverage=float(ok.mean()), unknown_rate=float(1 - ok.mean()), answered=ev.classification(yte[ok], pt[ok]),
                                    answered_ferplus=ev.classification(yte_fp[ok & okt], pt[ok & okt]))

    # ---- deploy graph + parity
    name = f"face-model-{ck['arch'].replace('_', '')}-{a.name}"
    d = Path(a.out_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    export_onnx(net, d / "face_fer.onnx")
    sess = ort.InferenceSession(str(d / "face_fer.onnx"), providers=["CPUExecutionProvider"])
    x = Xte[:512].astype(np.float32)[:, None]
    zo = sess.run(None, {"face": x})[0]
    net.cpu().float()
    with torch.no_grad():                                          # batches of 128: one 512-image CPU batch at res 112 can crash torch on a small laptop
        zt_cpu = np.concatenate([net(torch.from_numpy(x[i:i + 128])).numpy() for i in range(0, len(x), 128)])
    net.to(dev)
    out["parity"] = dict(max_abs_logit_diff_onnx_vs_torch_cpu=float(np.abs(zo - zt_cpu).max()), argmax_agreement=float((zo.argmax(1) == zt_cpu.argmax(1)).mean()),
                         max_abs_logit_diff_vs_gpu_fp16=float(np.abs(zo - zt[:512]).max()), argmax_agreement_vs_gpu_fp16=float((zo.argmax(1) == zt[:512].argmax(1)).mean()))
    zfull = np.concatenate([sess.run(None, {"face": Xte[i:i + 256].astype(np.float32)[:, None]})[0] for i in range(0, len(Xte), 256)])
    out["test_onnx"] = ev.classification(yte, P(zfull))

    # ---- latency / memory
    lat = dict(onnx_cpu_1thread_b1=ev.latency_ort(d / "face_fer.onnx", threads=1), onnx_cpu_2thread_b1=ev.latency_ort(d / "face_fer.onnx", threads=2),
               onnx_cpu_4thread_b1=ev.latency_ort(d / "face_fer.onnx", threads=4), onnx_cpu_2thread_b16=ev.latency_ort(d / "face_fer.onnx", threads=2, batch=16),
               **{f"torch_{'gpu_fp16' if dev.type == 'cuda' else 'cpu_fp32'}_b1": ev.latency_torch(net, dev, 1),
                  f"torch_{'gpu_fp16' if dev.type == 'cuda' else 'cpu_fp32'}_b32": ev.latency_torch(net, dev, 32)})
    if "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            lat["onnx_cuda_b1"] = ev.latency_ort(d / "face_fer.onnx", ("CUDAExecutionProvider", "CPUExecutionProvider"))
        except Exception as e:  # noqa: BLE001
            lat["onnx_cuda_b1"] = str(e)
    lat["onnx_size_mb"] = round((d / "face_fer.onnx").stat().st_size / 2**20, 2)
    lat["params_m"] = round(sum(p.numel() for p in net.parameters()) / 1e6, 3)
    out["efficiency"] = lat

    # ---- bundle
    torch.save(dict(state=ck["state"], arch=ck["arch"], res=ck["res"]), d / "face_fer.pt")
    cfg = dict(name=name, task="facial expression recognition (7 FER2013 classes) — an expression is what the face shows, not proof of a feeling",
               arch=ck["arch"], onnx="face_fer.onnx", classes=data.CLASSES, input=dict(name="face", shape=[None, 1, 48, 48], dtype="float32", range="0..255 gray",
               preprocessing_in_graph="bilinear 48->%d, /255, gray->3ch, ImageNet mean/std" % ck["res"]), net_res=ck["res"], temperature=T, prior_shift=shift, crop=CROP, unknown=unk,
               smoother=dict(tau_s=0.35, switch_margin=0.12, min_hold_s=0.3, strong=0.80, lost_s=1.5, min_conf=unk["min_conf"]),
               train=dict(args=ck["args"], run=a.run))
    (d / "config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
    card = dict(name=name, data="FER2013 (Kaggle train/test folders), 48x48 gray; labels: FER2013 one-hot mixed with FER+ votes per train args ferplus (Barsoum 2016, github.com/microsoft/FERPlus); val = 10% of train (stratified, duplicate-safe); test untouched",
                excluded_from_training=D["excluded"], metrics=dict(labels_for_fitting=a.labels, test_ferplus=out["test_ferplus"], test_ferplus_clean=out["test_ferplus_clean"], test=out["test"], test_clean=out["test_clean"], val=out["val"], val_ferplus=out["val_ferplus"]), calibration=out["test_calibration"],
                thresholds=unk, efficiency=lat, license_note="backbone: torchvision ImageNet-1K weights (BSD-3); data: FER2013 research use; model for research / demo",
                env=dict(python=platform.python_version(), torch=torch.__version__, onnxruntime=ort.__version__, gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else "none (CPU)"),
                known_limits=["FER2013 is static, low-resolution, tightly cropped, labels are noisy (human agreement ~65%)", "no sign-language video was used for training",
                              "output is an expression category, never an emotion diagnosis", "raised brows / open mouth are also sign-language grammar"])
    (d / "model_card.json").write_text(json.dumps(card, indent=1, ensure_ascii=False), encoding="utf-8")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"final_eval_{a.name}.json").write_text(json.dumps(out, indent=1))   # per model, so v1's final_eval.json is never overwritten
    print(json.dumps({k: out[k] for k in ("temperature", "thresholds", "parity")}, indent=1))
    for k in ("val", "test", "test_clean", "test_flip_tta", "test_onnx", "val_ferplus", "test_ferplus", "test_ferplus_clean"):
        print(k, {m: round(out[k][m], 4) for m in ("accuracy", "macro_f1", "weighted_f1", "uar")})
    print("ECE", out["test_calibration"]["ece"], "raw", out["test_calibration_raw"]["ece"], "unknown", out["test_with_unknown"]["unknown_rate"])
    print(json.dumps(lat, indent=1)); print("->", d)


if __name__ == "__main__":
    main()
