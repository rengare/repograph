#!/usr/bin/env python3
"""Consolidate the security workflow's results into one markdown report.

Reads whatever the scan jobs left behind — per-check cargo-deny statuses, the
clippy and zizmor SARIF files, and `cargo metadata` output — and writes a
single report. Missing inputs are reported as such rather than crashing, so a
job that failed early still produces a readable summary.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import subprocess
import sys

# SARIF severities, worst first. `none` is dropped from the tables entirely.
SEVERITIES = ["error", "warning", "note"]

STATUS_ICON = {
    "success": "✅",
    "failure": "❌",
    "cancelled": "⚪",
    "skipped": "⏭️",
}


def icon(status: str) -> str:
    return STATUS_ICON.get(status, "❔")


def read_sarif(path: pathlib.Path) -> tuple[collections.Counter, collections.Counter] | None:
    """Return (severity counts, per-rule counts), or None if unreadable."""
    try:
        sarif = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    by_severity: collections.Counter = collections.Counter()
    by_rule: collections.Counter = collections.Counter()
    for run in sarif.get("runs", []):
        # Rules carry the default severity; a result may override it.
        rule_default = {}
        driver = run.get("tool", {}).get("driver", {})
        for rule in driver.get("rules", []):
            level = rule.get("defaultConfiguration", {}).get("level")
            if level:
                rule_default[rule.get("id")] = level

        for result in run.get("results", []):
            rule_id = result.get("ruleId") or "(unknown)"
            level = result.get("level") or rule_default.get(rule_id) or "warning"
            if level == "none":
                continue
            by_severity[level] += 1
            by_rule[rule_id] += 1
    return by_severity, by_rule


def sarif_section(title: str, path: pathlib.Path, blocking: bool, top_n: int = 10) -> list[str]:
    out = [f"### {title}", ""]
    parsed = read_sarif(path)
    if parsed is None:
        out += [f"No SARIF produced (`{path.name}` missing or unreadable) — the scan did not complete.", ""]
        return out

    by_severity, by_rule = parsed
    total = sum(by_severity.values())
    note = "does not block merge" if not blocking else "blocks merge"
    if total == 0:
        out += [f"No findings. _({note})_", ""]
        return out

    counts = ", ".join(f"**{by_severity[s]}** {s}" for s in SEVERITIES if by_severity[s])
    out += [f"**{total}** findings — {counts}. _({note})_", ""]
    out += ["| Rule | Findings |", "| --- | ---: |"]
    for rule, n in by_rule.most_common(top_n):
        out.append(f"| `{rule}` | {n} |")
    if len(by_rule) > top_n:
        out.append(f"| _… {len(by_rule) - top_n} more rules_ | |")
    out.append("")
    return out


def deny_section(status_dir: pathlib.Path) -> list[str]:
    out = ["### Dependencies — cargo-deny", ""]
    # Each matrix job drops one `<check> <status>` file via upload-artifact.
    files = sorted(status_dir.glob("**/deny-*.txt")) if status_dir.is_dir() else []
    if not files:
        out += ["No cargo-deny statuses were recorded — the checks did not run.", ""]
        return out

    out += ["| Check | Result | Blocks merge |", "| --- | --- | --- |"]
    # `bans` is configured as a warning in deny.toml; the rest are errors.
    for f in files:
        try:
            check, status = f.read_text().split()
        except (OSError, ValueError):
            continue
        blocking = "no — warns only" if check == "bans" else "yes"
        out.append(f"| `{check}` | {icon(status)} {status} | {blocking} |")
    out.append("")
    return out


def inventory_section() -> list[str]:
    """Dependency inventory straight from cargo, mirroring what the SBOM records."""
    out = ["### Dependency inventory", ""]
    try:
        raw = subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--all-features"],
            capture_output=True, text=True, check=True,
        ).stdout
        meta = json.loads(raw)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        out += [f"Could not read `cargo metadata`: {exc}", ""]
        return out

    workspace = set(meta.get("workspace_members", []))
    packages = meta.get("packages", [])
    third_party = [p for p in packages if p["id"] not in workspace]

    licenses: collections.Counter = collections.Counter()
    unlicensed = []
    for p in third_party:
        expr = p.get("license")
        if expr:
            licenses[expr] += 1
        else:
            unlicensed.append(p["name"])

    out += [
        f"**{len(third_party)}** third-party crates across "
        f"**{len(packages) - len(third_party)}** workspace crates.",
        "",
        "| License expression | Crates |",
        "| --- | ---: |",
    ]
    for expr, n in licenses.most_common(10):
        out.append(f"| `{expr}` | {n} |")
    if len(licenses) > 10:
        out.append(f"| _… {len(licenses) - 10} more expressions_ | |")
    out.append("")

    if unlicensed:
        shown = ", ".join(f"`{n}`" for n in sorted(unlicensed)[:10])
        out += [
            f"⚠️ **{len(unlicensed)}** crates declare no license in their manifest: {shown}",
            "",
        ]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deny-status", type=pathlib.Path, default=pathlib.Path("artifacts"))
    ap.add_argument("--clippy-sarif", type=pathlib.Path, default=pathlib.Path("artifacts/clippy-sarif/clippy.sarif"))
    ap.add_argument("--zizmor-sarif", type=pathlib.Path, default=pathlib.Path("artifacts/zizmor-sarif/zizmor.sarif"))
    ap.add_argument("--gitleaks-result", default="unknown", help="job result string from needs.gitleaks.result")
    ap.add_argument("--deny-result", default="unknown", help="job result string from needs.deny.result")
    ap.add_argument("--commit", default="", help="commit SHA this report describes")
    ap.add_argument("--run-url", default="", help="URL of the workflow run")
    ap.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path("security-report.md"))
    args = ap.parse_args()

    blocking = {"cargo-deny": args.deny_result, "gitleaks": args.gitleaks_result}
    failed = [n for n, r in blocking.items() if r == "failure"]
    # Cancelled or skipped is not a pass: the check produced no verdict at all.
    inconclusive = [n for n, r in blocking.items() if r not in ("success", "failure")]

    lines = ["# Security report", ""]
    if failed:
        lines += [f"❌ **Blocking checks failed:** {', '.join(f'`{n}`' for n in failed)}", ""]
    elif inconclusive:
        names = ", ".join(f"`{n}`" for n in inconclusive)
        lines += [f"⚠️ **Blocking checks did not complete:** {names} — this is not a pass.", ""]
    else:
        lines += ["✅ **All blocking checks passed.**", ""]

    provenance = []
    if args.commit:
        provenance.append(f"Commit `{args.commit[:12]}`")
    if args.run_url:
        provenance.append(f"[workflow run]({args.run_url})")
    if provenance:
        lines += [" · ".join(provenance), ""]

    lines += ["## Blocking checks", ""]
    lines += deny_section(args.deny_status)
    scanned = " — full commit history scanned." if args.gitleaks_result in ("success", "failure") else "."
    lines += [
        "### Secrets — gitleaks", "",
        f"{icon(args.gitleaks_result)} **{args.gitleaks_result}**{scanned} _(blocks merge)_",
        "",
    ]

    lines += ["## Advisory checks", ""]
    lines += sarif_section("Static analysis — clippy", args.clippy_sarif, blocking=False)
    lines += sarif_section("Workflows — zizmor", args.zizmor_sarif, blocking=False)
    lines += ["Findings from both are also filed under the repository's **Security → Code scanning** tab.", ""]

    lines += ["## Inventory", ""]
    lines += inventory_section()
    lines += [
        "CycloneDX SBOMs for every workspace crate are produced by the "
        "**Build & Release** workflow, and the four shipped ones are attached "
        "to each release.",
        "",
    ]

    report = "\n".join(lines)
    args.output.write_text(report)
    print(report)
    # Never fail the report job itself; the scan jobs already own pass/fail.
    return 0


if __name__ == "__main__":
    sys.exit(main())
