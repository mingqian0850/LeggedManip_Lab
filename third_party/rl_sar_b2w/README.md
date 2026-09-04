# rl_sar B2W policy artifacts

This directory records, but does not vendor, the public Unitree B2W
`robot_lab` deployment policy from
[`fan-ziqi/rl_sar`](https://github.com/fan-ziqi/rl_sar).

## Reproducible source

- Repository: `https://github.com/fan-ziqi/rl_sar`
- Pinned revision: `376d42c9b128f963ab08579762d5a216a976ce39`
- Upstream model path: `policy/b2w/robot_lab/policy.pt`
- Upstream configuration paths:
  `policy/b2w/robot_lab/config.yaml` and `policy/b2w/base.yaml`
- Verified model size: 796,064 bytes
- Model SHA-256:
  `38155076408e8eccb22690c6c5be14bd1dcb9149245ca5e493308a9f6ff93b34`

The model blob at this revision is also the model shipped by upstream tags
`v4.0.0` and `v4.1.0`. The full commit is pinned because it additionally
records an explicit zero-valued default command in the B2W configuration.

Fetch and verify all artifacts with:

```bash
./third_party/rl_sar_b2w/fetch.sh
```

The files are downloaded to the ignored `artifacts/` directory. The helper
checks every file against `SHA256SUMS`, so a changed or incomplete upstream
response fails instead of silently replacing the expected policy.

## License and redistribution

The upstream repository declares Apache License 2.0 and contains no separate
license or model card next to this checkpoint. On that basis, this helper
treats the checked-in checkpoint and YAML metadata as covered by the root
Apache-2.0 license. The upstream `LICENSE` is downloaded and checksum-verified
alongside the model.

When redistributing the downloaded artifacts, include that `LICENSE`, retain
upstream attribution notices, and clearly mark any modified files. There is no
upstream `NOTICE` file at the pinned revision. This provenance record is an
engineering audit, not legal advice.
