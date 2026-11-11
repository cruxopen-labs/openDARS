import argparse
import json
from pathlib import Path
from .performance import Performance
from .report import analysis, generate, search_report, write_json
from .workload import ROOT


def main():
    parser = argparse.ArgumentParser(description="openTARS analytical DSE (not RTL simulation)")
    sub = parser.add_subparsers(dest="command", required=True)
    for cmd in ("calibrate", "analyze", "search", "report"):
        p = sub.add_parser(cmd)
        p.add_argument("--out", type=Path, default=ROOT / "reports")
        if cmd == "report":
            p.add_argument("--no-search", action="store_true")
        if cmd == "search":
            p.add_argument("--method", choices=("grid", "nsga2"), default="grid")
            p.add_argument("--target", choices=("fpga", "asic"), default="fpga")
            p.add_argument("--space", type=Path)
            p.add_argument("--population", type=int, default=32)
            p.add_argument("--generations", type=int, default=12)
            p.add_argument("--seed", type=int, default=7)
            p.add_argument("--position", type=int, default=250)
            p.add_argument("--tokens", type=int, default=20)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "report":
        print(json.dumps(generate(args.out, not args.no_search), indent=2))
        return
    perf = Performance().fit()
    if args.command == "calibrate":
        perf.save(args.out / "calibration.json")
        write_json(args.out / "validation.json", perf.validate())
    elif args.command == "analyze":
        write_json(args.out / "mapping.json", analysis(perf))
    else:
        result = search_report(perf, args.out, args.method, args.target, args.space,
                               args.population, args.generations, args.seed, args.position, args.tokens)
        print(json.dumps({k: result[k] for k in ("target", "method", "evaluated", "feasible")}))


if __name__ == "__main__":
    main()
