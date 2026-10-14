# Correction note: CNPq proposal (Faixa C)

**Date:** 2026-10-14

The proposal submitted to CNPq was prepared before the revision of the
`pgsg_2` article (CHEMOLAB-D-26-01046). Two statements in the submitted
version were later found to be unsupported and are corrected in the copy kept in
this folder (`pgsg_CNPq_C.tex` / `.pdf`).

## What changed

| Item | Submitted version | Corrected copy |
|---|---|---|
| NIR predictive advantage over PLS (H1) | Delta R2 = +0.244 (PGSGv2 0.8128 vs PLS 0.5684) | Delta R2 = -0.108 +/- 0.031 (PLS with cross-validated H; negative in all ten splits) |
| Raman predictive advantage (H1) | Delta R2 = +0.0246 | Delta R2 = +0.036 +/- 0.007 (positive in all ten splits) |
| `pgsg_1` (R2 = 0.806 vs 0.568, n = 1,448) | Listed as "submitted, under review", cited as evidence | Listed as discontinued; the claim relied on a fixed-complexity PLS baseline (H = 10) and is no longer cited |
| H1 status | Supported in NIR and Raman | Supported in Raman only |
| Status of `pgsg_2` | Pilot completed | Article accepted at Chemolab (CHEMOLAB-D-26-01046R1) |

## Why

1. The PLS baseline used a fixed number of latent variables (H = 10), inherited
   from an earlier dataset and never re-selected. With H chosen by 5-fold
   cross-validation (one-standard-error rule), PLS outperforms the gated model in NIR.
2. Both datasets contain replicate spectra of the same sample. The revised
   analysis keeps all spectra of a sample in the same partition.

H2, H3 and H4 (gate topology and prior reproducibility) are unaffected.

## References

- Accepted article: *Gate Topology and Reproducibility of Prior-Guided Spectral
  Gating Across Near-Infrared and Raman Spectroscopy*, Chemometrics and
  Intelligent Laboratory Systems (CHEMOLAB-D-26-01046R1).
- Archived code: https://doi.org/10.5281/zenodo.23370645
