"""Formula and Recency-B checks for the SDA-NG reference. Synthetic only."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "research" / "sda_ng"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"sda_ng_{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = load("reference")
hard_face = load("hard_face")


class SoftGateTests(unittest.TestCase):
    def test_frozen_constants_match_design(self):
        self.assertEqual(reference.TAU_FROZEN, 0.75)
        self.assertEqual(reference.TEMPERATURE, 0.10)
        self.assertEqual(reference.T_EPS, 1e-6)
        self.assertEqual(reference.QUANTILE_P, 0.30)
        self.assertEqual(reference.BETA, 0.05)

    def test_midpoint_and_bounds(self):
        g = reference.soft_gate(0.75, tau=0.75, temperature=0.10)
        self.assertAlmostEqual(float(g), 0.5, places=12)
        grid = np.linspace(-1.0, 1.0, 21)
        weights = reference.soft_gate(grid, tau=0.75, temperature=0.10)
        self.assertTrue(np.all(weights > 0.0))
        self.assertTrue(np.all(weights < 1.0))
        self.assertTrue(np.all(np.diff(weights) > 0.0))

    def test_temperature_is_clamped_at_eps(self):
        at_floor = reference.soft_gate(0.2, tau=0.0, temperature=0.0)
        at_eps = reference.soft_gate(0.2, tau=0.0, temperature=reference.T_EPS)
        at_negative = reference.soft_gate(0.2, tau=0.0, temperature=-3.0)
        np.testing.assert_allclose(at_floor, at_eps)
        np.testing.assert_allclose(at_negative, at_eps)
        wider = reference.soft_gate(0.2, tau=0.0, temperature=1.0)
        self.assertNotAlmostEqual(float(wider), float(at_eps), places=6)

    def test_documented_subthreshold_cosine_is_not_shut(self):
        # Prior LM path: max cosine ≈ 0.574 < frozen τ = 0.75.
        cos = 0.574
        T = 0.10
        expected = 1.0 / (1.0 + np.exp(-((cos - 0.75) / T)))
        g = float(reference.soft_gate(cos, tau=0.75, temperature=T))
        self.assertAlmostEqual(g, float(expected), places=12)
        self.assertGreater(g, 0.05)

    def test_hard_gate_is_inclusive_indicator(self):
        cos = np.array([0.749, 0.75, 1.0])
        np.testing.assert_array_equal(reference.hard_gate(cos, tau=0.75), [0.0, 1.0, 1.0])

    def test_residual_matches_formula_when_fast_state_is_wrong(self):
        memory = reference.RecencySDA(2, 1, capacity=1, beta=1.0, tau=0.75, temperature=0.10)
        memory.write(0, np.array([1.0, 0.0]), np.array([4.0]))
        memory.state[:] = 0.0
        out_h, info_h = memory.read(np.array([1.0, 0.0]), mode="hard", details=True)
        out_s, info_s = memory.read(np.array([1.0, 0.0]), mode="soft", details=True)
        self.assertAlmostEqual(float(info_h["cosine"]), 1.0, places=12)
        self.assertEqual(float(info_h["weight"]), 1.0)
        np.testing.assert_allclose(out_h, [4.0])
        g = float(reference.soft_gate(1.0, tau=0.75, temperature=0.10))
        self.assertAlmostEqual(float(info_s["weight"]), g, places=12)
        np.testing.assert_allclose(out_s, [g * 4.0])

    def test_quantile_is_linear_numpy_default(self):
        sample = np.array([0.1, 0.2, 0.4, 0.8])
        self.assertAlmostEqual(reference.calib_tau(sample, 0.3), float(np.quantile(sample, 0.3)))


class RecencyArchiveTests(unittest.TestCase):
    def test_lru_refresh_matches_recency_positions(self):
        memory = reference.RecencySDA(2, 1, capacity=2, beta=0.5, tau=0.75)
        writes = [(0, [1.0, 0.0], [1.0]),
                  (1, [0.0, 1.0], [2.0]),
                  (2, [1.0, 1.0], [3.0]),
                  (1, [0.0, 1.0], [9.0])]
        for key_id, key, value in writes:
            memory.write(key_id, np.asarray(key, dtype=float), np.asarray(value, dtype=float))
        self.assertEqual(memory.archived_ids(), [2, 1])
        positions = reference.recency_positions([0, 1, 2, 1], 2)
        self.assertEqual(positions.tolist(), [2, 3])
        out, info = memory.read(np.array([0.0, 1.0]), mode="hard", details=True)
        self.assertEqual(int(info["neighbor_id"]), 1)
        np.testing.assert_allclose(out, [9.0], atol=1e-8)

    def test_reconstructed_hard_face_keeps_late_keys(self):
        data = hard_face.make_hard_data(20250001)
        self.assertEqual(data["n"], 2048)
        self.assertEqual(data["U"], 1740)
        self.assertEqual(data["B"], 1024)
        self.assertEqual(data["n_rewrite_keys"] + data["n_distractor"], 716)
        self.assertLessEqual(data["n_rewrite_keys"] + data["n_distractor"], data["B"])
        memory = reference.RecencySDA(data["D"], data["V"], capacity=data["B"])
        for key_id, key, value in zip(data["key_ids"], data["keys"], data["values"]):
            memory.write(int(key_id), key, value)
        positions = reference.recency_positions(data["key_ids"], data["B"])
        self.assertEqual(memory.archived_ids(), [int(data["key_ids"][p]) for p in positions])
        archived = set(memory.archived_ids())
        self.assertTrue(set(int(i) for i in data["rewrite_ids"]) <= archived)
        self.assertTrue(set(int(i) for i in data["distractor_ids"]) <= archived)
        early_in = set(int(i) for i in data["early_ids"]) & archived
        self.assertEqual(
            len(early_in),
            data["B"] - data["n_rewrite_keys"] - data["n_distractor"],
        )
        # Latest rewrite value is the one stored, not the first write.
        by_id = dict(zip(memory.archived_ids(), range(memory.archive_size)))
        keys, values, ids = memory._matrices()
        del keys
        for key_id in data["rewrite_ids"]:
            slot = by_id[int(key_id)]
            last_pos = int(np.max(np.flatnonzero(data["key_ids"] == int(key_id))))
            np.testing.assert_allclose(values[slot], data["values"][last_pos])


if __name__ == "__main__":
    unittest.main()
