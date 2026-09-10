# Candidate distribution and publication handoff

Candidate: **sashimi-c 2.0.0rc1**; core dependency:
`sashimi-itamae>=0.2.0rc1,<0.3`. Python 3.11–3.13 is the required test matrix.

```bash
python -m pip install --find-links /path/to/reviewed-wheelhouse "sashimi-c==2.0.0rc1"
```

The candidate is not yet on PyPI. The wheelhouse contains the exact reviewed
core and variant wheels; dependency metadata uses versions, not Git URLs.
The optional uv source entry and lock are developer-only, pinning the component
CI's core revision. Artifact consumers do not require a checkout. To use uv
against a wheelhouse, pass `--no-sources --find-links /path/to/reviewed-wheelhouse`.

Build wheel and sdist from the clean exact candidate SHA. Record both hashes.
Unpack the sdist outside Git and rebuild its wheel without
`SASHIMI_C_SOURCE_REVISION`; the embedded source revision must survive.
Install both artifact forms in clean environments outside the source tree and
run representative catalogs/observables, provenance and full-family checks.
The review manifest is authoritative only after all five packages pass.

Main integration, changing repository visibility, public-index permission checks,
tags and uploads require a later explicit user instruction after peer review.
Recreate artifacts from the final integrated SHA and repeat affected checks plus
the exact family matrix. Publish core first, verify index installation, then
publish the variants and repeat the installed-artifact smoke. Index permission
and upload validation have not been performed by candidate preparation.

The inactive `release-workflow.yml.example` stays outside `.github/workflows`.
Before any later enabling, configure a protected PyPI environment with required
human reviewers and trusted publishing; manual dispatch alone is insufficient.
No upload workflow or release tag is enabled during this preparation.
