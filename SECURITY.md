# Security Policy

## Reporting a Vulnerability

**Do not open a public issue for security reports.**

Preferred channel: open a private advisory at
<https://github.com/xberg-io/sceptre/security/advisories/new>.

Alternative: email **<security@xberg.io>**.

Please include a description of the issue, steps to reproduce, affected versions, and your
preferred credit (or none). We acknowledge reports within **2 business days** and aim to
publish a fix within **14 days** for critical issues and **30 days** for others.

## Supported Versions

Security fixes target the latest release on `main` — currently the `0.5.x` line. Older
minor versions are not back-ported.

## Scope

In scope: the `sceptre` library, the `sceptre-cli` binary, and the MCP server surface.

Out of scope: ONNX Runtime, `tract`, `candle`, and other third-party dependencies (report
upstream and notify us). Model artifacts are fetched from Hugging Face and sha256-verified
against pins baked into the registry; a mismatch between a published artifact and its pin is
in scope and should be reported here.
