# Changelog

All notable changes to Sentinel are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CI matrix (ubuntu/windows × Python 3.10-3.13) with lint, test, coverage gate
- Coverage gate at 74% (measured 74.68% locally)
- Ruff lint configuration matching orvima's rule set
- `install.ps1` / `install.sh` one-line installers pulling latest GitHub Release
- Release workflow with trusted PyPI publishing, checksums, smoke install
- This CHANGELOG

### Changed
- README stats now measured (130 tests, 4 test files, ~2.1k source lines, 75% coverage gate)
- Version unified to 0.2.0 across pyproject.toml and src/sentinel/__init__.py
- Ruff auto-fixes applied across codebase (import sorting, collections.abc, zip strict, line length)

### Fixed
- Line-too-long in adversarial.py and policy.py (split strings)

## [0.1.0] - 2024-XX-XX

### Added
- Initial release: local structured classification for agent action gating
- Deterministic gate (tool baseline, risk concepts, weak net, transactional URL prefixes)
- Optional semantic classifier via Ollama (nomic-embed-text)
- Advisory mode by default — classifier records, does not decide
- `Classifier.ask()` for Noul/Choice/Score question types
- Calibration framework (Platt/Isotonic) with `calibrated` flag
- CLI: `sentinel gate`, `sentinel eval`, `sentinel bench`, `sentinel ask`, `sentinel score`, `sentinel doctor`
- 130 tests: gate correctness (76/76), adversarial set (42 cases), latency bench (p95 27.7ms)