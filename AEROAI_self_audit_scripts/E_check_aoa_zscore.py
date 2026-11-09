"""Script E - is the monotonicity penalty ever switched on?

The training script applies a threshold of 8.0 to the SCALED angle of attack
(column 3 of the scaled inputs). This prints the scaled value for each angle.
If the largest value is far below 8.0, the penalty never activates.

Usage:   python E_check_aoa_zscore.py                      (older export stats)
         python E_check_aoa_zscore.py path/to/cl_model_x_scaler.pkl
Needs:   numpy (and scikit-learn only if you pass a scaler file).
"""
import pickle
import sys

OLD_MEAN, OLD_SCALE = 3.6635, 4.8139   # AoA stats in models/models_meta.json
THRESHOLD = 8.0


def zscore(aoa, mean, scale):
    return (aoa - mean) / scale


def main(scaler_path=None):
    mean, scale = OLD_MEAN, OLD_SCALE
    if scaler_path:
        with open(scaler_path, "rb") as f:
            sc = pickle.load(f)
        mean, scale = float(sc.mean_[3]), float(sc.scale_[3])
    print(f"AoA scaler: mean {mean:.4f}, scale {scale:.4f}")
    print(f"{'AoA (deg)':>10} {'scaled value':>14}")
    for aoa in range(-6, 21, 2):
        print(f"{aoa:>10} {zscore(aoa, mean, scale):>14.3f}")
    top = zscore(20, mean, scale)
    print(f"\nlargest scaled angle in the API range (20 deg): {top:.2f}")
    if top < THRESHOLD:
        print(f"-> below the threshold {THRESHOLD}: a mask like (aoa > {THRESHOLD})"
              " applied to scaled angles is never true, so the penalty is inactive.")
    else:
        print("-> the threshold can be reached, so the penalty could activate.")
    degrees_needed = mean + THRESHOLD * scale
    print(f"A scaled value of {THRESHOLD} would need an angle of"
          f" {degrees_needed:.1f} degrees.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
