# Code Atlas — AIchat test installation

Source: user-supplied `Universal-Code-Atlas(1).zip` (Stage 66 according to included audit).

## Installation

1. Create branch `aichat1` in `KrakeCZ/aichat`.
2. Extract this ZIP **into the repository root**, preserving `code-atlas/` and `.github/workflows/code-atlas.yml`.
3. Commit only to `aichat1`.
4. Open GitHub Actions > `Code Atlas — aichat1 test export` > Run workflow; select branch `aichat1`.
5. Download the `code-atlas-aichat1-preview` artifact and inspect locally. **This workflow does not publish Pages or modify `main`.**

## Scope and limitations

- This is a staging package, not a certified production release.
- No application files are removed or modified by packaging.
- Includes original Python analyzers, offline graphics, vendored language assets and dependency archives, to avoid unverified removal of functional dependencies.
- Historical audit logs, development docs and test suite are excluded from this installation copy; they remain in the original ZIP.
- Running CI may install dependencies from PyPI; this is not a fully offline CI workflow.
- `--bundle-sources` publishes copies of indexed source files into the output. Use only with public projects and review outputs before deployment.
- GitHub Actions on forked or private repos may require manual enabling.
- Large binary assets are retained, including Kotlin compiler JAR (~57 MB), and GitHub upload limits should be considered.
- A successful GitHub Actions build is **not yet verified**; use the first test run to validate dependencies, platform assumptions and generated output.
