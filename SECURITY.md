# Security Policy

## Supported versions

repograph is pre-1.0. Only the latest release and the current `main` receive
fixes; there are no maintenance branches for older versions.

## Reporting a vulnerability

Please report privately rather than opening a public issue.

Use GitHub's private vulnerability reporting:
**[Security → Report a vulnerability](https://github.com/rengare/repograph/security/advisories/new)**

Include what you need to make the issue reproducible — affected component,
version or commit, steps, and impact. A proof of concept helps.

Expect an acknowledgement within a week. Once a fix is ready it ships in a
normal release and the advisory is published with credit, unless you would
rather stay anonymous.

## Scope

Everything in this repository: the `rkg` CLI, the `rkg-mcp` MCP server, the
`graphvisualizer` desktop viewer, and the `gv-web` browser build.

The places where untrusted input crosses a boundary, and so the most likely
sources of a real issue:

- **`crates/rkg-ingest`** parses arbitrary source files from a scanned
  repository with tree-sitter. A repository under an attacker's control is
  untrusted input to any tool that indexes it.
- **`crates/rkg-mcp`** is a JSON-RPC server on stdio, driven by an AI agent.
  Requests reaching it are not necessarily authored by the person running it.
- **`crates/rkg-cli`** shells out to `cargo install` from its `mcp install`
  subcommand, and writes agent configuration files into the user's project.
- **`crates/gv-web`** runs in the browser and loads graph files chosen by the
  viewer. Files are read client-side and never uploaded.

Out of scope: vulnerabilities in upstream crates with no repograph-specific
impact (report those to the crate, and to
[RustSec](https://github.com/rustsec/advisory-db)); and findings that require
the attacker to already control the machine running the tool.

## What we run

Every push and pull request is checked by
[`.github/workflows/security.yml`](.github/workflows/security.yml):

| Check | Tool | Blocking |
| --- | --- | --- |
| Dependency advisories, licenses, banned crates, source registries | `cargo-deny` ([`deny.toml`](deny.toml)) | yes |
| Rust static analysis | `clippy` → SARIF → code scanning | no |
| GitHub Actions workflow auditing | `zizmor` ([`zizmor.yml`](zizmor.yml)) | no |
| Secret scanning over full history | `gitleaks` | yes |

`cargo-deny` also runs weekly on a schedule, so a newly published RustSec
advisory surfaces without waiting for a commit.

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) additionally gates every
pull request on `cargo test --locked --workspace` and a
`wasm32-unknown-unknown` check, so nothing merges without compiling against the
committed `Cargo.lock`.

Every security run ends with a consolidated report
([`.github/scripts/security_report.py`](.github/scripts/security_report.py)):
per-check cargo-deny verdicts, clippy and zizmor findings broken down by rule,
gitleaks status, and a license inventory of the dependency tree. It is written
to the run's job summary and uploaded as a `security-report` artifact.

Dependency updates are proposed by Dependabot with a cooldown
([`.github/dependabot.yml`](.github/dependabot.yml)), so a freshly published
version is not adopted before the ecosystem has had a chance to flag it.

A CycloneDX SBOM is generated for every workspace crate on each build. The
SBOMs for the four shipped artifacts — `rkg-cli`, `rkg-mcp`, `gv-app`,
`gv-web` — are attached to each GitHub release alongside `checksums.txt`.
