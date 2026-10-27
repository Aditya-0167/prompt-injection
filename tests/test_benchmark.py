"""Integrity tests for the frozen benchmark."""
import hashlib, json, collections
import pytest
from helpers import REPO

CASES = json.loads((REPO / "benchmark" / "frozen_benchmark.json").read_text(encoding="utf-8"))


def test_hash_matches_the_recorded_hash():
    blob = json.dumps(CASES, sort_keys=True, ensure_ascii=False).encode("utf-8")
    assert hashlib.sha256(blob).hexdigest() == (REPO / "benchmark" / "benchmark.sha256").read_text().strip()


def test_size_and_balance():
    assert len(CASES) == 360
    assert collections.Counter(c["family"] for c in CASES) == {"severity": 120, "ioc": 120, "technique": 120}
    assert collections.Counter(c["gold_sev"] for c in CASES) == {"Medium": 120, "High": 120, "Critical": 120}
    assert set(collections.Counter(c["placement"] for c in CASES).values()) == {72}
    assert set(collections.Counter(c["strategy"] for c in CASES).values()) == {45}


def test_every_strategy_placement_pair_appears_three_times_per_family():
    for fam in ("severity", "ioc", "technique"):
        combos = collections.Counter((c["strategy"], c["placement"]) for c in CASES if c["family"] == fam)
        assert len(combos) == 40 and set(combos.values()) == {3}


@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_clean_twin_differs_in_exactly_one_line(case):
    clean, inj = case["report_clean"].split("\n"), case["report_inj_primary"].split("\n")
    assert len(clean) == len(inj)
    diff = [(a, b) for a, b in zip(clean, inj) if a != b]
    assert len(diff) == 1
    assert case["injection_primary"] in diff[0][1]
    assert case["injection_primary"] not in case["report_clean"]


def test_targets_are_valid():
    for c in CASES:
        if c["family"] == "ioc":
            assert c["target"] in c["gold_iocs"]
        if c["family"] == "technique":
            assert c["target"] != c["gold_tech"] and c["target"] not in c["report_clean"]
        if c["family"] == "severity":
            order = ["Low", "Medium", "High", "Critical"]
            assert order.index(c["target"]) < order.index(c["gold_sev"])


def test_only_documentation_addresses_and_reserved_domains_are_used():
    import re
    for c in CASES:
        for ioc in c["gold_iocs"]:
            if re.match(r"^\d+\.\d+\.\d+\.\d+$", ioc):
                assert ioc.startswith(("192.0.2.", "198.51.100.", "203.0.113."))
            else:
                assert ioc.endswith(".example")
