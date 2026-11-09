import numpy as np
from services.cst_features import naca4_to_cst, naca4_to_extra_feats

# Golden values copied from cd_val_locked.csv (NACA 1221, first row)
GOLD_1221 = [0.316460, 0.313053, 0.241122, 0.308138, -0.271313, -0.242008, -0.218571, -0.243147]


def test_matches_training_csv_golden_row():
    np.testing.assert_allclose(naca4_to_cst(1, 2, 21), GOLD_1221, atol=2e-6)


def test_symmetric_airfoil_is_mirror_and_linear_in_t():
    c12, c06 = naca4_to_cst(0, 0, 12), naca4_to_cst(0, 0, 6)
    np.testing.assert_allclose(c12[:4], -c12[4:], atol=1e-12)      # lower = -upper
    np.testing.assert_allclose(c12, 2 * c06, atol=1e-12)           # exactly linear in thickness
    assert abs(c12[0] - 0.167935) < 2e-6                            # NACA 0012 value in the CSV


def test_extra_feats_layout():
    f = naca4_to_extra_feats(2, 4, 12)
    assert f.shape == (10,) and f.dtype == np.float32
    assert abs(f[8] - 0.12) < 1e-6 and abs(f[9] - 1.1019 * 0.12 ** 2) < 1e-6
