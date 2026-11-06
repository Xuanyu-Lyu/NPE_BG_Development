"""Check feature handling for current and previously saved ACE models."""

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ace_model import VAR_REDUCTION, load_posterior


class ModelLoadingTests(unittest.TestCase):
    def test_internal_normalization_accepts_raw_features(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").write_text(
                json.dumps({
                    "var_feature": VAR_REDUCTION,
                    "feature_preprocessing": "sbi_z_score_x",
                })
            )
            with (model_dir / "posterior.pkl").open("wb") as handle:
                pickle.dump({"posterior": "placeholder"}, handle)

            raw = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
            loaded = load_posterior(model_dir)
            np.testing.assert_array_equal(loaded["scaler"].transform(raw), raw)

    def test_legacy_model_uses_saved_scaler(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").write_text(
                json.dumps({"var_feature": VAR_REDUCTION})
            )
            with (model_dir / "posterior.pkl").open("wb") as handle:
                pickle.dump({"posterior": "placeholder"}, handle)
            scaler = StandardScaler().fit([[1.0, 2.0], [3.0, 6.0]])
            joblib.dump(scaler, model_dir / "feature_scaler.pkl")

            loaded = load_posterior(model_dir)
            np.testing.assert_allclose(
                loaded["scaler"].transform([[3.0, 6.0]]), [[1.0, 1.0]]
            )


if __name__ == "__main__":
    unittest.main()
