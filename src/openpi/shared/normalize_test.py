import numpy as np

import openpi.shared.normalize as normalize


def test_normalize_update():
    arr = np.arange(12)

    stats = normalize.RunningStats()
    for i in range(0, len(arr), 3):
        stats.update(arr[i : i + 3])
    results = stats.get_statistics()

    assert np.allclose(results.mean, np.mean(arr))
    assert np.allclose(results.std, np.std(arr))


def test_serialize_deserialize():
    stats = normalize.RunningStats()
    stats.update(np.arange(12))

    norm_stats = {"test": stats.get_statistics()}
    norm_stats2 = normalize.deserialize_json(normalize.serialize_json(norm_stats))
    assert np.allclose(norm_stats["test"].mean, norm_stats2["test"].mean)
    assert np.allclose(norm_stats["test"].std, norm_stats2["test"].std)


def test_align_norm_stats_to_model_dim():
    ns = normalize.NormStats(
        mean=np.ones(3),
        std=np.ones(3) * 0.5,
        q01=-np.ones(3),
        q99=np.ones(3),
    )
    out = normalize.align_norm_stats_to_model_dim(
        {"a": ns}, model_action_dim=5, robot_action_dim=3
    )
    m = out["a"].mean
    s = out["a"].std
    assert m.shape == (5,)
    assert np.allclose(m[:3], 1.0)
    assert np.allclose(s[:3], 0.5)
    assert np.allclose(m[3:], 0.0)
    assert np.allclose(s[3:], 1.0)
    assert out["a"].q01 is not None and out["a"].q99 is not None
    assert np.allclose(out["a"].q01[3:], -1.0)
    assert np.allclose(out["a"].q99[3:], 1.0)
