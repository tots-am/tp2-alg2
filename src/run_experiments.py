"""Executor dos experimentos.

  python -m src.run_experiments heuristics   # rápido: heurísticas + features + profiling
  python -m src.run_experiments bnb          # lento: B&B paralelo (retomável)

Decisões para reduzir o tempo total de parede:
  * B&B em C++ (subprocesso) — o GIL não importa; paralelismo por processo.
  * 1 worker por núcleo FÍSICO, fixado com taskset (hyperthreading distorce tempos).
  * Ordem LPT: jobs maiores (N alto, cold start) primeiro, para os de 30 min
    começarem cedo e os curtos preencherem as lacunas no fim.
  * Cada job grava sua linha ao terminar -> a execução é retomável (pula feitos).
  * Timeout dentro do binário: NA preserva nós expandidos/podados até o corte.
  * Heurísticas rodadas UMA vez na fase 'heuristics'; o B&B reusa o tour salvo
    e soma o tempo da heurística ao custo total (trade-off do warm-start).
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import statistics
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")       # NumPy/BLAS sem threads internas
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from .tsp import bnb, features, heuristics, instances  # noqa: E402

RES = os.path.join(instances.ROOT, "results")
TOURS = os.path.join(RES, "tours")
STRATS = list(heuristics.STRATEGIES)


# ----------------------------------------------------------------------------
# Fase 1: heurísticas, features e profiling por sub-rotina
# ----------------------------------------------------------------------------
def phase_heuristics(reps: int, scaling: bool):
    os.makedirs(TOURS, exist_ok=True)
    tests = instances.test_set()
    feats = [features.extract(i) for i in tests]
    _write_csv(os.path.join(RES, "features.csv"), feats)
    print(f"features: {len(feats)} instâncias")

    sets = [("test", tests)]
    if scaling:
        sets.append(("scaling", instances.scaling_set()))

    rows = []
    for setname, insts in sets:
        for inst in insts:
            for s in STRATS:
                times, sub_times = [], {}
                for _ in range(reps):                       # passada de tempo
                    tour, cost, t, prof = heuristics.run_strategy(s, inst)
                    times.append(t)
                    for k, v in prof.times.items():
                        sub_times.setdefault(k, []).append(v)
                _, _, _, pm = heuristics.run_strategy(s, inst, memory=True)   # passada de memória
                row = {"set": setname, "instance": inst.name, "n": inst.n, "strategy": s,
                       "cost": cost, "time_median": statistics.median(times),
                       "time_min": min(times), "reps": reps,
                       "peak_mem_bytes": max(pm.peaks.values(), default=0)}
                for k, v in sub_times.items():
                    row[f"t_{k}"] = statistics.median(v)
                for k, v in pm.peaks.items():
                    row[f"mem_{k}"] = v
                rows.append(row)
                if setname == "test":
                    with open(os.path.join(TOURS, f"{inst.name}__{s}.json"), "w") as f:
                        json.dump({"cost": cost, "time": statistics.median(times), "tour": tour}, f)
            print(f"[{setname}] {inst.name:14s} n={inst.n:3d} ok", flush=True)
    _write_csv(os.path.join(RES, "heuristics.csv"), rows)


# ----------------------------------------------------------------------------
# Fase 2: Branch-and-Bound paralelo
# ----------------------------------------------------------------------------
BNB_FIELDS = ["instance", "n", "strategy", "bound", "time_limit", "status", "cost",
              "initial_ub", "heur_time", "bnb_time", "total_time", "nodes_expanded",
              "nodes_pruned", "leaves", "bound_calls", "ub_updates", "max_depth",
              "root_lb", "root_lb_hk", "time_hk_root", "time_bound_est", "time_to_best",
              "maxrss_kb", "cpu", "finished_at"]

_CPU = None


def _init_worker(q):
    global _CPU
    _CPU = q.get()


def _job(args):
    name, k, strat, bound, tl = args
    inst = instances.load(name, k)
    with open(os.path.join(TOURS, f"{inst.name}__{strat}.json")) as f:
        w = json.load(f)
    r = bnb.solve(inst.D, w["cost"], w["tour"], bound=bound, time_limit=tl, cpu=_CPU)
    row = {k_: r.get(k_) for k_ in BNB_FIELDS if k_ in r}
    row.update(instance=inst.name, n=inst.n, strategy=strat, bound=bound, time_limit=tl,
               status="NA" if r["timed_out"] else "optimal",
               heur_time=w["time"], bnb_time=r["time_total"],
               total_time=w["time"] + r["time_total"], cpu=_CPU,
               finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    return row, r["tour"]


def phase_bnb(workers: int, time_limit: float, bounds: list[str], max_n: int | None,
              only: list[str] | None, strats: list[str]):
    out = os.path.join(RES, "bnb.csv")
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            done = {(r["instance"], r["strategy"], r["bound"]) for r in csv.DictReader(f)}

    jobs = []
    for name, k in instances.TEST_SET:
        iname = f"{name}_k{k}" if k else name
        n = k or instances.load(name).n
        if (max_n and n > max_n) or (only and iname not in only):
            continue
        for b in bounds:
            for s in strats:
                if (iname, s, b) not in done:
                    jobs.append(((name, k, s, b, time_limit), n))
    # LPT: maiores primeiro; dentro do mesmo N, cold start (mais lento) primeiro
    jobs.sort(key=lambda j: (-j[1], STRATS.index(j[0][2])))
    jobs = [j for j, _ in jobs]
    if not jobs:
        print("nada a fazer (todos os jobs já estão em bnb.csv)")
        return

    cpus = bnb.physical_cpus()[:workers]
    workers = len(cpus)
    print(f"{len(jobs)} jobs, {workers} workers nas CPUs {cpus}, limite {time_limit:.0f}s; "
          f"pior caso ~{len(jobs) * time_limit / workers / 3600:.1f} h")
    q = mp.Queue()
    for c in cpus:
        q.put(c)

    new = not os.path.exists(out)
    t0 = time.time()
    with open(out, "a", newline="") as f, \
            open(os.path.join(RES, "bnb_tours.jsonl"), "a") as ft, \
            mp.Pool(workers, initializer=_init_worker, initargs=(q,)) as pool:
        wr = csv.DictWriter(f, fieldnames=BNB_FIELDS)
        if new:
            wr.writeheader()
        for i, (row, tour) in enumerate(pool.imap_unordered(_job, jobs), 1):
            wr.writerow(row)
            f.flush()
            ft.write(json.dumps({"instance": row["instance"], "strategy": row["strategy"],
                                 "bound": row["bound"], "cost": row["cost"], "tour": tour}) + "\n")
            ft.flush()
            print(f"[{i}/{len(jobs)} +{(time.time() - t0) / 60:5.1f}min] {row['instance']:14s} "
                  f"{row['strategy']:12s} {row['bound']:5s} {row['status']:7s} cost={row['cost']} "
                  f"exp={row['nodes_expanded']} pod={row['nodes_pruned']} t={row['bnb_time']:.2f}s",
                  flush=True)


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("heuristics")
    h.add_argument("--reps", type=int, default=5)
    h.add_argument("--no-scaling", action="store_true")
    b = sub.add_parser("bnb")
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--time-limit", type=float, default=1800.0)
    b.add_argument("--bounds", default="hk", help="ex.: hk  ou  hk,mst,naive")
    b.add_argument("--strategies", default=",".join(STRATS))
    b.add_argument("--max-n", type=int)
    b.add_argument("--only", help="lista de instâncias separadas por vírgula")
    a = ap.parse_args(argv)
    if a.cmd == "heuristics":
        phase_heuristics(a.reps, not a.no_scaling)
    else:
        phase_bnb(a.workers, a.time_limit, a.bounds.split(","), a.max_n,
                  a.only.split(",") if a.only else None, a.strategies.split(","))


if __name__ == "__main__":
    sys.exit(main())
