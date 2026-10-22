import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import plot  # noqa: E402


def test_plot_and_table(tmp_path):
    run = tmp_path / "outputs" / "base" / "checkpoint-2"
    run.mkdir(parents=True)
    hist = [{"step": s, "reward": 0.1 * s, "kl": 0.01 * s} for s in range(1, 4)]
    (run / "trainer_state.json").write_text(json.dumps({"log_history": hist}))
    res = tmp_path / "results" / "base"
    res.mkdir(parents=True)
    (res / "metrics.json").write_text(
        json.dumps({"accuracy": 0.5, "format_rate": 0.9, "mean_chars": 300.0})
    )

    figs = tmp_path / "figs"
    figs.mkdir()
    plot.plot_curves(["base"], tmp_path / "outputs", figs)
    assert (figs / "training_curves.png").exists()
    table = plot.results_table(["zero_shot", "base"], tmp_path / "results")
    assert "| base | 50.0% | 90.0% | 300 |" in table
    assert "zero_shot" not in table  # missing runs are skipped
