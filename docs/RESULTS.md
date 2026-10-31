# Evaluation summary

This page records the high-level outcomes associated with the codebase. Detailed inputs and per-scenario records are not included in the public repository.

## Evaluations

| Evaluation | Scenario groups | Throughput change versus A3-style baseline | Handover-rate change versus baseline |
| --- | ---: | ---: | ---: |
| Same-area reserved route evaluation | 639 | +8.86% to +9.99% | -36.08% to -53.00% |
| Pokhara simulation transfer | 540 | +3.66% to +3.85% | -23.68% to -47.10% |

Each range spans the three frozen learned policies. The comparison uses matched scenarios and one configured A3-style reference. Pokhara uses real road geometry with synthetic cell layouts, not field measurements.

## Interpretation limits

- Same-area test route identifiers were reserved, but their road segments overlap with training data.
- Same-area short-stay proportions increased for all three learned policies.
- Pokhara transfer gains vary across layouts and seeds.
- The three training seeds do not characterize training variability reliably.
- No matched no-message-passing model establishes that the graph architecture caused the observed differences.
- A post hoc load-aware baseline search did not qualify any candidate for confirmatory comparison.
- The observations include simulator-wide radio/load estimates and known-route lookahead.
- The results do not establish real-network performance, field validity, or deployment readiness.

The reported bootstrap intervals are conditional on frozen policies and sampled scenarios. They do not establish population-level significance across cities, network deployments, or training seeds.

## Source of records

The values above were transcribed from the project's verified private evaluation summary. Raw route inputs, cell layouts, checkpoints, case-level records, and bootstrap inputs are not included in this repository, so the values cannot be independently regenerated from a clean clone.
