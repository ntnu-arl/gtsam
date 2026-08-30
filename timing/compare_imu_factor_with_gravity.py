#!/usr/bin/env python3
"""Run timeImuFactorWithGravity from several builds and tabulate the results.

Each build is given as ``--bin label=path``. The binaries are run round-robin
for ``--rounds`` rounds (order reversed on odd rounds so thermal drift cancels),
optionally pinned to one core with ``taskset``. Every cell is the median across
rounds of the harness's own per-run median, and the tables follow the layout of
GTSAM PR #2765: earlier columns first, then signed improvement percentages,
where improvement = (earlier - later) / earlier * 100.

Example:
    python3 timing/compare_imu_factor_with_gravity.py --cpu 3 --rounds 5 \\
        --bin before=/path/before/timing/timeImuFactorWithGravity \\
        --bin after=/path/after/timing/timeImuFactorWithGravity \\
        --output results.md -- --samples 10000 --calls 2000 --warmups 20 --repetitions 41
"""

import argparse
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path


def parse_harness_output(text):
    """Return ({backend: (pim, combined|None)}, {(backend, factor): (error, linearize)})."""
    integration, factors, section = {}, {}, None
    for line in text.splitlines():
        if line.startswith("Preintegration"):
            section = "integration"
            continue
        if line.startswith("Factor evaluation"):
            section = "factors"
            continue
        parts = line.split()
        if not parts or parts[0] == "backend":
            continue
        if section == "integration" and len(parts) == 3:
            combined = None if parts[2] == "-" else float(parts[2])
            integration[parts[0]] = (float(parts[1]), combined)
        elif section == "factors" and len(parts) == 4:
            factors[(parts[0], parts[1])] = (float(parts[2]), float(parts[3]))
    if not integration or not factors:
        raise ValueError("could not parse harness output:\n" + text)
    return integration, factors


def run_once(binary, harness_args, cpu):
    command = ([f"taskset", "-c", str(cpu)] if cpu is not None else []) + [binary] + harness_args
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"{' '.join(command)} failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def median_of(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def improvement(earlier, later):
    return (earlier - later) / earlier * 100.0


def fmt(value):
    return "-" if value is None else f"{value:.1f}"


def environment_line(cpu, extra):
    model = "unknown CPU"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    governor = None
    if cpu is not None:
        try:
            governor = open(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor").read().strip()
        except OSError:
            pass
    parts = [platform.system(), model]
    if cpu is not None:
        parts.append(f"pinned to core {cpu}" + (f" (governor {governor})" if governor else ""))
    if extra:
        parts.append(extra)
    return ", ".join(parts)


TABLE_NOTES = {
    "standard": "Cost of `integrateMeasurement` per IMU sample for a plain `PreintegratedImuMeasurementsT<backend>`, "
                "the object held by `ImuFactor`, `ImuFactor2` and their gravity variants alike. Gravity does not enter "
                "preintegration, so this is not specific to the gravity factors; it is included because a factor's "
                "total cost per interval is preintegration plus its evaluations.",
    "combined": "The same for `PreintegratedCombinedMeasurementsT<backend>`, which additionally propagates the bias "
                "random walk (15x15 covariance) and is held by `CombinedImuFactor` and its gravity variants.",
    "error": "Cost of one factor's residual at the linearization point without any Jacobians "
             "(`factor.unwhitenedError(values)`): the path taken by `graph.error()` and line search.",
    "linearize": "What the optimizer pays per factor per iteration: `factor.linearize(values)` computes the residual, "
                 "every Jacobian block (including the gravity block), whitens, and builds the `JacobianFactor`. Each "
                 "gravity factor is listed below the plain factor it wraps.",
    "overhead": "Marginal cost of making gravity a variable: each gravity factor's `linearize` time minus that of the "
                "plain factor with the same backend, in ns and as a percentage of the plain factor. A flat row across "
                "columns means the compared builds neither helped nor hurt the gravity-specific work.",
}


def build_tables(labels, rounds_integration, rounds_factors):
    """rounds_* : {label: [parsed dict per round]}. Returns markdown lines."""
    first, last = labels[0], labels[-1]
    middle = labels[-2] if len(labels) >= 3 else None

    def header(first_column):
        cols = [first_column] + [f"{label} (ns)" for label in labels]
        cols.append(f"Improvement {first}→{last} %")
        if middle:
            cols.append(f"Improvement {middle}→{last} %")
        return "| " + " | ".join(cols) + " |\n|" + "---|" + "---:|" * (len(cols) - 1)

    def row(name, values):
        cells = [name] + [fmt(values[label]) for label in labels]
        if values[first] is None or values[last] is None:
            cells += ["-"] + (["-"] if middle else [])
        else:
            cells.append(f"{improvement(values[first], values[last]):+.1f}")
            if middle:
                cells.append(f"{improvement(values[middle], values[last]):+.1f}" if values[middle] is not None else "-")
        return "| " + " | ".join(cells) + " |"

    def cell(label, key, kind, index):
        source = rounds_integration if kind == "integration" else rounds_factors
        return median_of([r[key][index] for r in source[label]])

    lines = []
    backends = list(rounds_integration[first][0].keys())
    lines += ["### Standard PIM preintegration (ns per sample)", "", TABLE_NOTES["standard"], "", header("Backend")]
    for backend in backends:
        lines.append(row(backend, {label: cell(label, backend, "integration", 0) for label in labels}))
    lines += ["", "### Combined PIM preintegration (ns per sample)", "", TABLE_NOTES["combined"], "", header("Backend")]
    for backend in backends:
        values = {label: cell(label, backend, "integration", 1) for label in labels}
        if any(v is not None for v in values.values()):
            lines.append(row(backend, values))

    factor_keys = list(rounds_factors[first][0].keys())
    lines += ["", "### Factor error only (`unwhitenedError`, no Jacobians; ns per call)", "", TABLE_NOTES["error"], "", header("Backend / factor")]
    for key in factor_keys:
        lines.append(row(f"{key[0]} `{key[1]}`", {label: cell(label, key, "factors", 0) for label in labels}))
    lines += ["", "### Factor `linearize` (error, Jacobians, whitening; ns per call)", "", TABLE_NOTES["linearize"], "", header("Backend / factor")]
    for key in factor_keys:
        lines.append(row(f"{key[0]} `{key[1]}`", {label: cell(label, key, "factors", 1) for label in labels}))

    # Gravity-awareness overhead: linearize cost above the plain sibling factor.
    lines += ["", "### Gravity-awareness overhead in `linearize` over the plain sibling (ns per call)", "", TABLE_NOTES["overhead"], "",
              "| Backend / factor | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    siblings = {"ImuFactorWithGravity": "ImuFactor", "ImuFactor2WithGravity": "ImuFactor2",
                "CombinedImuFactorWithGravity": "CombinedImuFactor"}
    for key in factor_keys:
        family = key[1].split("<")[0]
        if family not in siblings:
            continue
        cells = []
        for label in labels:
            gravity = cell(label, key, "factors", 1)
            plain = cell(label, (key[0], siblings[family]), "factors", 1)
            cells.append("-" if gravity is None or plain is None else f"{gravity - plain:.0f} ({(gravity / plain - 1) * 100:.1f}%)")
        lines.append(f"| {key[0]} `{key[1]}` | " + " | ".join(cells) + " |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bin", action="append", required=True, metavar="LABEL=PATH",
                        help="a build of timeImuFactorWithGravity; give at least two, earliest first")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--cpu", type=int, default=None, help="pin every run to this core with taskset")
    parser.add_argument("--env", default="", help="extra text for the environment line (compiler, flags, commits)")
    parser.add_argument("--output", type=Path, default=None, help="write the markdown report here as well as stdout")
    parser.add_argument("--save-raw", type=Path, default=None, help="write every parsed round as JSON")
    parser.add_argument("harness_args", nargs="*", help="arguments passed through to the harness after --")
    args = parser.parse_args()

    labels, binaries = [], {}
    for spec in args.bin:
        label, _, path = spec.partition("=")
        if not path:
            parser.error(f"--bin expects LABEL=PATH, got {spec!r}")
        labels.append(label)
        binaries[label] = path
    if len(labels) < 2:
        parser.error("give at least two --bin entries")

    rounds_integration = {label: [] for label in labels}
    rounds_factors = {label: [] for label in labels}
    raw = {label: [] for label in labels}
    for round_index in range(args.rounds):
        order = labels if round_index % 2 == 0 else list(reversed(labels))
        for label in order:
            output = run_once(binaries[label], args.harness_args, args.cpu)
            integration, factors = parse_harness_output(output)
            rounds_integration[label].append(integration)
            rounds_factors[label].append(factors)
            raw[label].append(output)
            print(f"round {round_index + 1}/{args.rounds}: {label} done", file=sys.stderr)

    report = [f"Environment: {environment_line(args.cpu, args.env)}.",
              f"Medians over {args.rounds} interleaved rounds; harness arguments: `{' '.join(args.harness_args) or '(defaults)'}`.",
              ""] + build_tables(labels, rounds_integration, rounds_factors)
    text = "\n".join(report) + "\n"
    print(text)
    if args.output:
        args.output.write_text(text)
    if args.save_raw:
        args.save_raw.write_text(json.dumps({"labels": labels, "binaries": binaries,
                                             "harness_args": args.harness_args, "outputs": raw}, indent=1))


if __name__ == "__main__":
    main()
