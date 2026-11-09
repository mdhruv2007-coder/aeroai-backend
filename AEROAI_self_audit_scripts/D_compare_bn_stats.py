"""Script D - did BatchNorm statistics change between two saved model files?

Usage:   python D_compare_bn_stats.py FILE_A.pt FILE_B.pt
Example: python D_compare_bn_stats.py models/rans_adapter_weights.pt \
                                     models/aero_ai_v2.0_multifidelity.pt
Prints the largest difference in each BatchNorm running mean and variance and
the change in num_batches_tracked. A difference of 30 batches (or non-zero
mean/variance changes) would mean something ran the adapter in training mode
between the two saves (suspect: Test 7).
Needs:   torch. Loads with weights_only=True. Use --trusted only for files
that you created yourself.
"""
import sys

import numpy as np

SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def unwrap(obj):
    """Find the state dict inside a checkpoint."""
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "adapter_state_dict"):
            if isinstance(obj.get(key), dict):
                return obj[key]
    return obj


def bn_buffers(sd):
    return {k: np.asarray(v, dtype=float) for k, v in sd.items()
            if k.endswith(SUFFIXES)}


def match_key(key, other_keys):
    for k in other_keys:
        if k == key or k.endswith("." + key) or key.endswith("." + k):
            return k
    return None


def compare(a, b):
    """Returns (rows, unmatched). Each row: (name, d_mean/var or d_batches)."""
    rows, unmatched = [], []
    for ka, va in a.items():
        kb = match_key(ka, list(b))
        if kb is None:
            unmatched.append(ka)
            continue
        rows.append((ka, kb, float(np.max(np.abs(va - b[kb])))))
    return rows, unmatched


def main(path_a, path_b, trusted=False):
    import torch
    load = lambda p: unwrap(torch.load(p, map_location="cpu",
                                       weights_only=not trusted))
    to_np = lambda sd: {k: (v.cpu().numpy() if hasattr(v, "cpu") else v)
                        for k, v in sd.items()}
    a = bn_buffers(to_np(load(path_a)))
    b = bn_buffers(to_np(load(path_b)))
    print(f"BatchNorm buffers: {len(a)} in A, {len(b)} in B")
    rows, unmatched = compare(a, b)
    changed = 0
    for ka, kb, diff in rows:
        flag = "" if diff == 0 else "  <-- differs"
        changed += diff != 0
        print(f"{ka:<50} max|diff| {diff:.3e}{flag}")
    if unmatched:
        print(f"\n{len(unmatched)} buffers in A had no match in B (names differ)."
              " Check the key names printed above.")
    if changed:
        print(f"\n{changed} buffers differ: the files do not hold the same"
              " normalisation statistics.")
    else:
        print("\nAll matched buffers are identical. Either nothing changed, or"
              " both files were saved after the same step. Script C is the real"
              " test.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--trusted"]
    if len(args) != 2:
        sys.exit(__doc__)
    main(args[0], args[1], trusted="--trusted" in sys.argv)
