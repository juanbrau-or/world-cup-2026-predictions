# Static dashboard

The static dashboard is generated from the public `predictions-data` branch only. It does not read
`.env`, raw provider payloads, Parquet files, models, ledgers or private artifacts.

Build locally from a checked-out data branch:

```bash
uv run wc2026 site build --data-root <predictions-data-root> --output-root site-dist
```

The build is deterministic for the same input files. It writes:

- HTML pages under `site-dist/`;
- `assets/styles.css` and `assets/site.js`;
- `build-manifest.json`;
- `checksum-report.json`;
- `page-coverage.json`;
- `simulation-fallback-audit.json`.

The output directory is ignored and should not be committed. GitHub Pages receives it as an Actions
artifact from `.github/workflows/deploy-dashboard.yml`.

## Public data contract

Required files:

- `manifest.json`;
- `latest.csv`;
- `latest.json`;
- `upcoming.md`;
- `prospective_scorecard.json`;
- `prospective_matches.csv`.

Optional sections are rendered when present:

- `shadow/*` for the contextual challenger in shadow mode;
- `simulation/*` for tournament Monte Carlo summaries.

The builder fails when checksums do not match, probabilities do not sum to one, timestamps are not
UTC, official and shadow predictions are mixed, secret-like content appears, or disallowed paths
such as raw data, `.env`, Parquet or model files exist in the data root.
