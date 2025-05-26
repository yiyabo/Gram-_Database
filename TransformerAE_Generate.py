#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Strict generation script (v2)
----------------------------
* Compatible with the Global‑Feature‑VAE training pipeline (run_*/ directories).
* Adds temperature / top‑k sampling and multiple z‑sampling modes.
* Workflow:  ➜  build latent batch  ➜  decode ➜  dedup ➜  optional k‑mer split trim ➜  AMP classifier filter ➜  save FASTA/TXT.
"""
import os, glob, argparse, pickle, subprocess
from datetime import datetime
from typing import List, Dict

import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ────────────────────────────────
# GPU auto‑select + TF config
# ────────────────────────────────
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    try:
        q = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"], text=True
        ).strip().split("\n")
        best = min(range(len(q)), key=lambda i: int(q[i] or 0))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best)
        print(f"[AutoGPU] selected GPU:{best}  mem_used:{q[best]} MiB")
    except Exception as e:
        print("[AutoGPU] fallback:", e)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)

# ────────────────────────────────
# CLI
# ────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", help="VAE run_* directory; default = latest run", default=None)
parser.add_argument("-n", "--num_samples", type=int, default=10000)
parser.add_argument(
    "--mode", choices=["random", "meanstd", "global"], default="meanstd",
    help=(
        "latent sampling mode: "
        "random -> N(0,I); "
        "meanstd -> N(mean,std) clipped (+/-1σ); "
        "global -> sample around global_features centers"
    ),
)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=0, help="top‑k sampling (0 = greedy/sample over all)")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--threshold", type=float, default=0.5, help="AMP classifier pass threshold")
parser.add_argument("--classifier_dir", default="model")
parser.add_argument("--output_dir", default="output")
args = parser.parse_args()

# ────────────────────────────────
# Locate newest run if not provided
# ────────────────────────────────
if args.run_dir is None:
    runs = sorted(glob.glob("model/vae_nature/run_*"))
    if not runs:
        raise FileNotFoundError("No VAE runs found; train the model first.")
    run_dir = runs[-1]
    print("[Auto] using latest run:", run_dir)
else:
    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run_dir 不存在: {run_dir}")

# ────────────────────────────────
# Load VAE & metadata
# ────────────────────────────────
params: Dict[str, str] = {}
with open(os.path.join(run_dir, "params.txt")) as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            params[k] = v
latent_dim  = int(params["latent_dim"])
max_len     = int(params["max_len"])
split_times = int(params["split_times"])
overlap     = int(params.get("overlap", 2))
print(f"latent_dim={latent_dim}, seq_len={max_len}, split_len={split_times}, overlap={overlap}")

decoder = tf.keras.models.load_model(os.path.join(run_dir, "decoder.h5"), compile=False)
with open(os.path.join(run_dir, "tokenizer.pkl"), "rb") as f:
    tok_dict = pickle.load(f)
rev_tok = {v: k for k, v in tok_dict.items()}

mean_z = std_z = None
if os.path.exists(os.path.join(run_dir, "z_samples.npy")):
    zs = np.load(os.path.join(run_dir, "z_samples.npy"))
    mean_z, std_z = zs.mean(0), zs.std(0)

global_feats = None
if args.mode == "global":
    gf_path = os.path.join(run_dir, "global_features.npy")
    if os.path.exists(gf_path):
        global_feats = np.load(gf_path)
        print(f"Loaded {len(global_feats)} global feature centroids.")

# ────────────────────────────────
# Sampling helpers
# ────────────────────────────────

def sample_latent(n: int) -> np.ndarray:
    if args.mode == "random":
        return np.random.randn(n, latent_dim)
    if args.mode == "meanstd":
        if mean_z is None:
            raise RuntimeError("mean/std not found; retrain or use --mode random")
        z = np.random.normal(mean_z, std_z, (n, latent_dim))
        return np.clip(z, mean_z - std_z, mean_z + std_z)
    # global
    if global_feats is None:
        raise RuntimeError("global_features.npy missing; cannot use --mode global")
    idx   = np.random.choice(len(global_feats), n)
    base  = global_feats[idx]
    noise = 0.35 * np.random.randn(n, latent_dim)
    return base + noise

def decode_logits(logits: np.ndarray) -> List[str]:
    """Greedy decoder: at each timestep pick the token with highest prob."""
    seqs: List[str] = []
    for step_probs in logits:                      # shape = [seq_len, vocab]
        # ── 贪婪选 token ─────────────────────────
        ids = step_probs.argmax(axis=1)            # vectorised argmax
        # ── k‑mer id → 氨基酸序列 ────────────────
        kmers = [rev_tok.get(i, "") for i in ids if i]
        if not kmers:
            seqs.append(""); continue
        seq = kmers[0]
        for km in kmers[1:]:
            seq += km[-1]                          # overlap=2, k=3
        seqs.append(seq)
    return seqs

# def decode_logits(logits: np.ndarray) -> List[str]:
#     """
#     Sampling decoder
#     --------------------------------------------------
#     • 温度 T : 把 logits / T 再 soft‑max
#     • top‑k  : 只在最大概率的前 k 个 token 中采样
#                (k == 0 时表示不裁剪，等同 nucleus≈1.0)
#     • 遇到 PAD id==0 立即停止，得到可变长度序列
#     """
#     T  = args.temperature
#     k  = args.top_k
#     seqs: List[str] = []

#     for step_probs in logits:                 # shape = [seq_len, vocab]
#         ids = []
#         for vec in step_probs:
#             # 1) 温度
#             if T != 1.0:
#                 vec = vec / T

#             # 2) top‑k 裁剪（可选）
#             if k and k > 0:
#                 topk_idx = np.argpartition(vec, -k)[-k:]
#                 mask = np.full_like(vec, -np.inf)
#                 mask[topk_idx] = 0.0
#                 vec = vec + mask

#             # 3) soft‑max → 概率
#             p = tf.nn.softmax(vec).numpy()

#             # 4) multinomial 采样
#             idx = np.random.choice(len(p), p=p)

#             # 5) PAD / EOS (id==0) 立即结束
#             if idx == 0:
#                 break
#             ids.append(idx)

#         # ---- k‑mer → 氨基酸串 ----
#         kmers = [rev_tok.get(i, "") for i in ids]
#         if not kmers:
#             seqs.append("")
#             continue
#         seq = kmers[0]
#         for km in kmers[1:]:
#             seq += km[-1]                     # overlap=2, k=3
#         seqs.append(seq)
#     return seqs


# ────────────────────────────────
# Generate
# ────────────────────────────────
print(f"Sampling {args.num_samples} latent vectors …")
all_z    = sample_latent(args.num_samples)
all_seqs = []
for i in tqdm(range(0, len(all_z), args.batch_size), desc="Decoding"):
    logits = decoder.predict(all_z[i:i+args.batch_size], verbose=0)
    all_seqs.extend(decode_logits(logits))
all_seqs = [s for s in all_seqs if s]
print(f"Got {len(all_seqs)} non‑empty sequences; mean len = {np.mean(list(map(len, all_seqs))):.1f}")

# Deduplicate
uniq = list(dict.fromkeys(all_seqs))
print(f"Deduplicated: {len(uniq)} sequences")

# ────────────────────────────────
# Classifier filtering
# ────────────────────────────────
clf_p = os.path.join(args.classifier_dir, "filter_best_model.h5")
tok_p = os.path.join(args.classifier_dir, "tokenized_dict_filter.pkl")
len_p = os.path.join(args.classifier_dir, "max_length_filter.txt")
if not (os.path.exists(clf_p) and os.path.exists(tok_p) and os.path.exists(len_p)):
    raise RuntimeError("Classifier assets not found in --classifier_dir")
classifier = tf.keras.models.load_model(clf_p, compile=False)
with open(tok_p, "rb") as f:
    clf_tok = pickle.load(f)
clf_maxlen = int(open(len_p).read().strip())

tokd = [[clf_tok.get(aa.upper(), 0) for aa in s] for s in uniq]
padded = pad_sequences(tokd, maxlen=clf_maxlen, padding='post')
probs = classifier.predict(padded, batch_size=args.batch_size, verbose=0).flatten()
mask  = probs >= args.threshold
passed = [s for s, keep in zip(uniq, mask) if keep]
print(f"Classifier threshold {args.threshold} → passed {mask.sum()}/{len(mask)} ({mask.mean():.2%})")

# ────────────────────────────────
# Save
# ────────────────────────────────
os.makedirs(args.output_dir, exist_ok=True)
ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
root = f"gen_{os.path.basename(run_dir)}_{ts}_T{args.temperature}_k{args.top_k}"
fa_p = os.path.join(args.output_dir, root + ".fasta")
txt_p = os.path.join(args.output_dir, root + ".txt")

with open(fa_p, "w") as fa, open(txt_p, "w") as tx:
    for i, seq in enumerate(passed, 1):
        prob = probs[mask][i-1]          # 取对应概率，别忘了
        fa.write(f">SEQ_{i}\n{seq}\n")
        tx.write(seq + "\n")

print(f"Saved {len(passed)} sequences → {txt_p}")