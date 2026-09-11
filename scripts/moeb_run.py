"""Run one model's share of the MOEB evaluation pool.

Work is derived, not assigned: the script computes the tasks this model is
eligible for, subtracts whatever already exists in the public results repo, and
runs the remainder. Two people running it on the same model never collide, and
nobody needs a spreadsheet.

    # what would I run?  (no GPU needed)
    python scripts/moeb_run.py --model Qwen/Qwen2.5-Omni-3B --dry-run

    # take one slice of the work; run N shards in parallel across machines
    python scripts/moeb_run.py --model Qwen/Qwen2.5-Omni-3B --shard 0/4

    # when done, open a PR against the results repo
    python scripts/moeb_run.py --model Qwen/Qwen2.5-Omni-3B --submit

Eligibility rule: a model may run a task iff the task's declared modalities are
a subset of the model's. That keeps specialists on their own slice and gives the
omni models the full matrix, without a hand-maintained list.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time

import mteb

logger = logging.getLogger("moeb_run")

# Pin the evaluator. Results produced by different mteb versions are not
# comparable, and a campaign that mixes them cannot be pooled afterwards.
REQUIRED_MTEB_VERSION = "2.20.12"

# The text anchor: every model also runs this, so each has one comparable text
# number without re-running the whole 649-task text suite.
TEXT_ANCHOR = "MTEB(eng, v2)"


def _shard(items: list[str], index: int, total: int) -> list[str]:
    """Deterministic content-based sharding: stable across machines and reruns."""
    if total <= 1:
        return items
    keep = []
    for name in items:
        h = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        if h % total == index:
            keep.append(name)
    return keep


def eligible_tasks(model_meta, include_text_anchor: bool) -> list[str]:
    model_mods = set(model_meta.modalities or [])
    names = []
    for task in mteb.get_tasks(exclude_beta=False):
        mods = set(task.metadata.modalities or [])
        if not mods or not mods <= model_mods:
            continue
        if mods == {"text"}:
            continue  # text-only handled by the anchor below
        names.append(task.metadata.name)
    if include_text_anchor and "text" in model_mods:
        try:
            names += [t.metadata.name for t in mteb.get_benchmark(TEXT_ANCHOR).tasks]
        except Exception:  # noqa: BLE001 - benchmark name may move between versions
            logger.warning("could not resolve %s; skipping text anchor", TEXT_ANCHOR)
    return sorted(dict.fromkeys(names))


def already_done(model_name: str, task_names: list[str], offline: bool) -> set[str]:
    if offline:
        return set()
    try:
        results = mteb.load_results(models=[model_name], tasks=task_names,
                                    only_main_score=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read results repo (%s); running everything", exc)
        return set()
    done = set()
    for model_result in results:
        for task_result in model_result.task_results:
            if task_result.mteb_version == REQUIRED_MTEB_VERSION:
                done.add(task_result.task_name)
    return done


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--shard", default="0/1", help="i/N, e.g. 2/8")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--num-proc", type=int, default=4,
                   help="dataloader workers; media decode is the bottleneck")
    p.add_argument("--dry-run", action="store_true", help="list the work and exit")
    p.add_argument("--offline", action="store_true",
                   help="skip the results-repo check (run everything eligible)")
    p.add_argument("--submit", action="store_true",
                   help="after running, open a PR against the results repo")
    p.add_argument("--no-text-anchor", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="cap task count (testing)")
    p.add_argument("--only-modality", default=None,
                   help="restrict to tasks declaring this modality, e.g. audio")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if mteb.__version__ != REQUIRED_MTEB_VERSION:
        sys.exit(
            f"mteb {mteb.__version__} != pinned {REQUIRED_MTEB_VERSION}.\n"
            "Results from different versions are not comparable; install the "
            "pinned version before running."
        )

    index, total = (int(x) for x in args.shard.split("/"))
    meta = mteb.get_model_meta(args.model)

    pool = eligible_tasks(meta, include_text_anchor=not args.no_text_anchor)
    if args.only_modality:
        keep = {
            t.metadata.name for t in mteb.get_tasks(exclude_beta=False)
            if args.only_modality in (t.metadata.modalities or [])
        }
        pool = [t for t in pool if t in keep]
    done = already_done(args.model, pool, args.offline)
    todo = _shard([t for t in pool if t not in done], index, total)
    if args.limit:
        todo = todo[: args.limit]

    print(f"model      : {args.model}  ({','.join(sorted(meta.modalities or []))})")
    print(f"eligible   : {len(pool)} tasks")
    print(f"already in results repo @ {REQUIRED_MTEB_VERSION}: {len(done)}")
    print(f"this shard : {len(todo)} tasks  (shard {index}/{total})")
    if args.dry_run:
        for name in todo:
            print("  ", name)
        return 0
    if not todo:
        print("nothing to do")
        return 0

    cache = mteb.ResultCache()
    failures = []
    for n, task_name in enumerate(todo, 1):
        started = time.time()
        try:
            task = mteb.get_tasks(tasks=[task_name])[0]
            mteb.evaluate(
                mteb.get_model(args.model, device=args.device),
                [task],
                cache=cache,
                encode_kwargs={"batch_size": args.batch_size},
                num_proc=args.num_proc,
                overwrite_strategy="only-missing",
                raise_error=False,
            )
            logger.info("[%d/%d] %s ok (%.1f min)",
                        n, len(todo), task_name, (time.time() - started) / 60)
        except Exception as exc:  # noqa: BLE001 - one bad task must not end the run
            failures.append((task_name, f"{type(exc).__name__}: {exc}"))
            logger.exception("[%d/%d] %s FAILED", n, len(todo), task_name)

    print(f"\ndone: {len(todo) - len(failures)}/{len(todo)} tasks")
    for name, err in failures:
        print(f"  FAILED {name}: {err}")

    if args.submit:
        response = cache.submit_results(args.model, create_pr=True)
        print("submission:", response)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
