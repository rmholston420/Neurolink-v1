# Security Policy

## Supported Versions

Only the latest commit on `main` is actively maintained and receives security fixes.

| Version | Supported          |
| ------- | ------------------ |
| `main`  | :white_check_mark: |
| older   | :x:                |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, report vulnerabilities privately via one of these channels:

1. **GitHub Private Vulnerability Reporting** — use the [Security Advisories](../../security/advisories/new) tab in this repository (recommended).
2. **Email** — send a description to the repository owner through their GitHub profile contact. Expect an acknowledgement within **72 hours** and a resolution timeline within **14 days** for critical issues.

## What to Include

A useful vulnerability report includes:
- A clear description of the issue and its potential impact
- Steps to reproduce, or a minimal proof-of-concept
- The affected component(s) and version/commit SHA
- Any suggested mitigations, if known

## Scope

This policy applies to the Neurolink-v1 backend (`backend/`), DSP pipeline (`dsp/`), hardware adapters (`hardware/`), and API routers (`routers/`). Frontend assets and documentation files are out of scope unless a vulnerability in them can be exploited to attack the backend.

## Disclosure Policy

Once a fix is merged to `main` and released, a GitHub Security Advisory will be published with full details and credit to the reporter.
