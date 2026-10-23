"""Run: python -m examples.quickstart (after pip install -e .)."""
import numpy as np
from summer_delta_attention import SummerDeltaAttention, kda_proxy_state, full_kv_attention


def main():
    rng = np.random.default_rng(20261023)
    n, d_key, d_value = 128, 32, 16
    keys = rng.normal(size=(n, d_key))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    values = rng.normal(size=(n, d_value))

    sda = SummerDeltaAttention(d_key, d_value, capacity=64,
                               beta=0.05, threshold=0.75, seed=7)
    for key, value in zip(keys, values):
        sda.write(key, value)

    kda_state = kda_proxy_state(keys, values)
    ids = rng.integers(n, size=64)
    query = keys[ids]
    targets = values[ids]
    sda_predictions = np.stack([sda.read(q) for q in query])
    kda_predictions = query @ kda_state
    full_predictions = np.stack([full_kv_attention(q, keys, values) for q in query])
    mse = lambda pred: float(np.mean((pred - targets) ** 2))

    print("Synthetic exact-key association recall; lower MSE is better")
    print(f"SDA v5-Gated (reservoir archive): {mse(sda_predictions):.4f}")
    print(f"KDA-style untrained proxy:        {mse(kda_predictions):.4f}")
    print(f"Full-KV sharp softmax reference:  {mse(full_predictions):.4f}")
    print(f"SDA archive: {sda.archive_size}/{n}; array payload: "
          f"{sda.bytes_of_state_and_archive / 1024:.1f} KiB")
    print("This is not a trained KDA/Kimi Linear benchmark.")


if __name__ == "__main__":
    main()
