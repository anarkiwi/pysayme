# CI and releasing

## CI

`.github/workflows/ci.yml` runs on pushes to `main` and on pull requests:

- **lint**: `black --check` and `pylint` over `src` and `tests`.
- **test**: `pytest`, with xdist and coverage of at least 85%, on Python 3.11–3.14.
  pysayme's own `[tool.pysayme]` table makes the suite gate pysayme against its
  own dependencies.
- **build**: builds the sdist and wheel, runs `twine check --strict`, and runs the
  wheel's console script.

Dependabot updates `uv.lock` and the workflow actions weekly.

## Releasing to PyPI

Versions come from git tags (hatch-vcs).

`.github/workflows/release.yml` builds and publishes when a GitHub release is
published. It uses PyPI trusted publishing (OIDC), so no API token is stored.

One-time setup:

1. On PyPI, add a trusted publisher for the project `pysayme`. If the project does
   not exist yet, add it as a pending publisher. Use these values:
   - owner: `anarkiwi`
   - repository: `pysayme`
   - workflow: `release.yml`
   - environment: `pypi`
2. On GitHub, create the environment `pypi` under Settings → Environments.
   Optionally, restrict it to tags and require reviewers.

To release, tag and publish a release:

```
git tag v0.1.0 && git push origin v0.1.0
gh release create v0.1.0 --generate-notes
```
