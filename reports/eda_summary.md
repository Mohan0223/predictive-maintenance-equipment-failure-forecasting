# Exploratory data analysis summary

The charts use only the first 80% of rows ordered by UDI, matching the training partition prepared in Phase 3.

- Training records: 8,000
- No-failure records: 7,700
- Failure records: 300
- Recorded failure rate: 3.75%
- Dataset provenance: synthetic AI4I benchmark data, not factory telemetry.
- Vibration and pressure are absent, so no sensor-vs-failure values are available for those fields.

## Median measured features by recorded outcome

| Feature | No failure | Failure |
|---|---:|---:|
| Air temperature [K] | 300.500 | 301.900 |
| Process temperature [K] | 310.300 | 310.600 |
| Rotational speed [rpm] | 1507.000 | 1363.500 |
| Torque [Nm] | 39.800 | 53.600 |
| Tool wear [min] | 107.000 | 153.000 |

## Failure rate by product type

| Type | Records | Failure rate |
|---|---:|---:|
| L | 4,816 | 4.22% |
| M | 2,402 | 3.16% |
| H | 782 | 2.69% |

These are descriptive summaries of a synthetic dataset. Differences and correlations are not causal evidence.
