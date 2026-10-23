"""Keep the committed benchmark artifacts, figure script and README in sync.

The original release shipped a summary CSV that experiment.py could not have
produced (different method names, KDA identical to DeltaNet), so figures.py
failed and the README's KDA number was absent from the data.
"""
import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "research" / "summary_results.csv"
README = ROOT / "README.md"

# Must match NEEDED in research/figures.py.
REQUIRED_METHODS = {
    "Full KV softmax (sharp)", "KDA channel-gated proxy",
    "No-decay DeltaNet (calibrated beta)", "Indexed KV (zero fallback)",
    "SDA v5-Gated (original)", "SDA v5-Gated (calibrated)",
    "Zero predictor", "Archive softmax (same retained KV)",
}

# (method, README text that must appear, rounded mean, rounded SEM)
HEADLINE = [
    ("SDA v5-Gated (calibrated)", "0.4721 ± 0.0099", 0.4721, 0.0099),
    ("Indexed KV (zero fallback)", "0.4962 ± 0.0101", 0.4962, 0.0101),
    ("KDA channel-gated proxy", "0.9336 ± 0.0098", 0.9336, 0.0098),
    ("Zero predictor", "0.9786 ± 0.0098", 0.9786, 0.0098),
]


def _main_rows():
    with CSV.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["n"] == "512" and float(r["noise"]) == 0.0
                and float(r["retention"]) == 0.5]
    return {r["method"]: r for r in rows}


class ArtifactConsistencyTests(unittest.TestCase):
    def test_summary_csv_has_every_method_the_figures_need(self):
        self.assertFalse(REQUIRED_METHODS - set(_main_rows()))

    def test_kda_proxy_is_distinct_from_no_decay_delta(self):
        rows = _main_rows()
        self.assertNotEqual(rows["KDA channel-gated proxy"]["mean_mse"],
                            rows["No-decay DeltaNet (calibrated beta)"]["mean_mse"])

    def test_readme_headline_numbers_match_csv(self):
        rows, text = _main_rows(), README.read_text(encoding="utf-8")
        for method, shown, mean, sem in HEADLINE:
            with self.subTest(method=method):
                self.assertEqual(round(float(rows[method]["mean_mse"]), 4), mean)
                self.assertEqual(round(float(rows[method]["sem"]), 4), sem)
                self.assertIn(shown, text)


if __name__ == "__main__":
    unittest.main()
