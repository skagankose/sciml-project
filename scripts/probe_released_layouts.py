"""Can any input layout make a released checkpoint behave like a one-step model? Data-free (synthetic input).

    HF_HOME=... python scripts/probe_released_layouts.py <dir with a 512x128 synthetic dataset> [--model tfno]

A working one-step model's output tracks its last input frame. This tries channel order (t f) / (f t), an
x/y transpose, flips of either axis and all 24 field orders, and prints the best combinations by their
worst-field correlation with the last input frame (decisions D20). The test is necessary, not sufficient: the
released CNextU-Net gives 0.86 with the correct layout but up to 0.94 with wrong field orders (its skip
connections pass the input through), so a low score everywhere rules layouts out; a high one confirms nothing.
"""
import itertools
import sys
from pathlib import Path

import torch
from einops import rearrange

from rbbench.config import load_config
from rbbench.data import full_trajectory_dataset, list_trajectories, load_trajectory
from rbbench.released import fetch, load_released

torch.set_num_threads(8)
d = Path(sys.argv[1]) / "datasets/rayleigh_benard"
fd, _ = full_trajectory_dataset(d, "valid", 4)
traj = load_trajectory(fd, list_trajectories(d, "valid")[0], d).float()
x = traj[None, 4:8]; last = x[0, -1]
MODEL = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--model" else "tfno"
net = load_released(MODEL, load_config(MODEL, "main", "full"), fetch(MODEL), paper_forward=False).net

def corr(a, b):
    return [torch.corrcoef(torch.stack([a[..., c].flatten(), b[..., c].flatten()]))[0, 1].item() for c in range(4)]

def run(xx, chan, transpose, flips, perm):
    xx = xx[..., list(perm)]
    for ax in flips: xx = xx.flip(ax)
    if transpose: xx = xx.transpose(2, 3)
    pat = "b t x y f -> b (t f) x y" if chan == "tf" else "b t x y f -> b (f t) x y"
    out = net(rearrange(xx, pat))                      # [1, F, X', Y']
    if transpose: out = out.transpose(2, 3)
    for ax in flips: out = out.flip(ax - 1)            # output has no time axis
    out = rearrange(out, "b f x y -> b x y f")[0]
    inv = [perm.index(i) for i in range(4)]
    return out[..., inv]

results = []
with torch.no_grad():
    for chan, tr, flips in itertools.product(("tf", "ft"), (False, True), ((), (2,), (3,), (2, 3))):
        c = corr(run(x, chan, tr, flips, (0, 1, 2, 3)), last)
        results.append((min(c), f"chan={chan} transpose={tr} flip_axes={flips}", c))
    for perm in itertools.permutations(range(4)):
        c = corr(run(x, "tf", False, (), perm), last)
        results.append((min(c), f"field order {perm}", c))
results.sort(reverse=True)
print("best 6 of", len(results), "(min over fields of corr(output, last input frame); a working model gives ~0.9+):")
for m, name, c in results[:6]:
    print(f"  {m:+.3f}  {name:42s} {[round(v, 3) for v in c]}")
