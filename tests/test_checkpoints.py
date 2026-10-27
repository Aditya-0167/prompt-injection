"""Tests on the checkpoint ZIPs and the merged raw outputs."""
import collections, gzip, importlib.util, json, zipfile
import pytest
from helpers import REPO

ZIPS = sorted((REPO / "checkpoints").glob("cti_*.zip"))
EXPECTED_TIERS = {"core": 3360, "adaptive": 840, "defense_aware": 720, "sampled": 720}


def read_rows(zpath):
    with zipfile.ZipFile(zpath) as z:
        name = [n for n in z.namelist() if n.endswith("raw/rows.jsonl")][0]
        return [json.loads(line) for line in z.open(name)]


@pytest.fixture(scope="module")
def merged():
    outputs = collections.defaultdict(set)
    meta = {}
    for z in ZIPS:
        for r in read_rows(z):
            outputs[r["key"]].add(r["raw_output"])
            meta[r["key"]] = r
    return outputs, meta


def test_checkpoints_present():
    assert len(ZIPS) == 11


def test_union_of_checkpoints_equals_the_released_raw_outputs(merged):
    outputs, _ = merged
    with gzip.open(REPO / "raw" / "rows.jsonl.gz", "rt", encoding="utf-8") as f:
        released = {json.loads(line)["key"] for line in f}
    assert set(outputs) == released and len(released) == 29280


def test_rows_per_model_and_tier(merged):
    _, meta = merged
    cnt = collections.Counter((r["model"], r["tier"]) for r in meta.values())
    primary = ["Qwen3-4B", "Phi-4-mini", "Qwen2.5-3B", "Granite-3.3-2B"]
    for m in primary:
        for tier, n in EXPECTED_TIERS.items():
            assert cnt[(m, tier)] == n, (m, tier)
    for m in ["SmolLM2-1.7B", "Qwen2.5-1.5B"]:
        assert cnt[(m, "core")] == 3360 and cnt[(m, "adaptive")] == 0


def test_cross_session_regeneration_conflicts_are_the_documented_four(merged):
    outputs, meta = merged
    conflicts = [k for k, v in outputs.items() if len(v) > 1]
    assert len(conflicts) == 4
    assert {meta[k]["model"] for k in conflicts} == {"Phi-4-mini"}


def test_merge_tool_deduplicates_and_checks_the_benchmark(tmp_path):
    spec = importlib.util.spec_from_file_location("merge_tool", REPO / "code" / "restore_and_merge_checkpoints.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows, dup, conflict, active = mod.restore_and_merge([str(z) for z in ZIPS], dest=str(tmp_path / "dest"),
                                                         tmp=str(tmp_path / "work"))
    assert len(rows) == 29280 and dup > 0 and active == {120}
