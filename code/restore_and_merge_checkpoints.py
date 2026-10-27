# ======================================================================
# CELL 1 -- RESTORE + MERGE CHECKPOINTS (run this BEFORE the main script)
#
# 1. Run this cell. When the file picker opens, select ALL the checkpoint ZIPs you have
#    (from any Colab account, any session). Hold Ctrl/Shift to pick many at once.
# 2. It merges them into /content/CTITRIAGE_V3, drops duplicate rows, and checks that
#    the benchmark is identical in every ZIP (it refuses to merge otherwise).
# 3. Then run the main script (Cell 2) unchanged. It continues where you left off.
#
# MULTIPLE ACCOUNTS: to split the remaining work, set MY_MODELS below on each account,
# e.g. account A: "Phi-4-mini,Qwen2.5-3B"   account B: "Granite-3.3-2B,Qwen3-4B"
# Leave it "" to let one account do everything. Afterwards, upload each account's
# final ZIP into a fresh session with this same cell to merge them, then run the
# main script with MY_MODELS = "" and RUN_STATS_ONLY = True to produce the final tables.
# ======================================================================
MY_MODELS = ""            # e.g. "Granite-3.3-2B,Qwen2.5-3B"   ("" = all models)
RUN_STATS_ONLY = False    # True = only recompute statistics/figures from the merged rows (no generation)

import os, io, json, glob, shutil, zipfile, hashlib, collections


def restore_and_merge(zip_paths, dest="/content/CTITRIAGE_V3", tmp="/content/_restore"):
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    parts = []
    for i, zp in enumerate(zip_paths):
        out = os.path.join(tmp, f"z{i}")
        with zipfile.ZipFile(zp) as z:
            z.extractall(out)
        roots = [r for r in glob.glob(out + "/*") if os.path.isdir(r)]
        root = next((r for r in roots if os.path.exists(r + "/raw/rows.jsonl")), None)
        if root is None:
            raise SystemExit(f"{os.path.basename(zp)} has no raw/rows.jsonl: wrong file?")
        parts.append((os.path.basename(zp), root))

    hashes = {open(r + "/benchmark/benchmark.sha256").read().strip() for _, r in parts}
    if len(hashes) != 1:
        raise SystemExit(f"Benchmark hashes differ across ZIPs ({len(hashes)} different): refusing to merge.")
    actives = {json.load(open(r + "/benchmark/active_subset.json"))["active_n_per_family"]
               for _, r in parts if os.path.exists(r + "/benchmark/active_subset.json")}
    if len(actives) > 1:
        raise SystemExit(f"Different active_n_per_family across ZIPs: {actives}; refusing to merge.")

    rows, seen, dup, conflict = [], {}, 0, 0
    for name, root in parts:
        for line in open(root + "/raw/rows.jsonl", encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            k = r["key"]
            if k in seen:
                dup += 1
                if seen[k] != r["raw_output"]:
                    conflict += 1
                continue
            seen[k] = r["raw_output"]
            rows.append(r)

    if os.path.exists(dest):
        shutil.rmtree(dest)
    for d in ["benchmark", "raw", "statistics", "figures", "tables", "environment", "logs", "checkpoints"]:
        os.makedirs(f"{dest}/{d}")
    best = max(parts, key=lambda p: sum(1 for _ in open(p[1] + "/raw/rows.jsonl", encoding="utf-8")))[1]
    for f in glob.glob(best + "/benchmark/*"):
        shutil.copy(f, f"{dest}/benchmark/")
    rows.sort(key=lambda r: r.get("timestamp", ""))
    with open(f"{dest}/raw/rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # keep the EARLIEST preregistration documents (never overwrite history)
    for doc in ["PREREGISTRATION.md", "PREREG_AMENDMENT_1.md", "PREREG_AMENDMENT_2.md", "PREREG_AMENDMENT_3.md"]:
        cands = [r + "/" + doc for _, r in parts if os.path.exists(r + "/" + doc)]
        if cands:
            earliest = min(cands, key=lambda p: open(p).read().split("Written:")[1][:32])
            shutil.copy(earliest, f"{dest}/{doc}")
    for name, root in parts:
        for lg in glob.glob(root + "/logs/*.log"):
            shutil.copy(lg, f"{dest}/logs/restored_{name}.log")
    shutil.rmtree(tmp, ignore_errors=True)
    return rows, dup, conflict, actives


if __name__ == "__main__" or "get_ipython" in dir():
    try:
        from google.colab import files
        up = files.upload()
        paths = []
        for n, data in up.items():
            with open(f"/content/{n}", "wb") as f:
                f.write(data)
            paths.append(f"/content/{n}")
    except ImportError:
        import sys
        paths = sys.argv[1:]
    assert paths, "No ZIP selected"
    rows, dup, conflict, actives = restore_and_merge(paths)
    cnt = collections.Counter((r["model"], r["tier"]) for r in rows)
    print(f"\nMerged {len(paths)} ZIP(s): {len(rows)} unique rows | duplicate rows skipped: {dup} | "
          f"duplicates whose output DIFFERED: {conflict} | cases/family: {sorted(actives)}")
    for (m, t), n in sorted(cnt.items()):
        print(f"   {m:16s} {t:14s} {n}")
    os.environ["CTITRIAGE_ONLY_MODELS"] = MY_MODELS
    if RUN_STATS_ONLY:
        os.environ["CTITRIAGE_GEN_STOP"] = "0"
    print(f"\nCTITRIAGE_ONLY_MODELS='{MY_MODELS}'  stats_only={RUN_STATS_ONLY}")
    print("Now run the main script (Cell 2) unchanged.")
