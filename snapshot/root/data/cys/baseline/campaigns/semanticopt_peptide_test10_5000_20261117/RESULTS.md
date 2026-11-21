# SemanticOPT continuation: same ten peptide tasks

All ten tasks reached 4,000 distinct scored proposals, excluding 100 initial observations. The first 1,000 saved observations per task were preserved exactly. Lower predicted MIC is better.

| Task | MIC at 1,000 | MIC at 4,000 | Further improvement |
|---|---:|---:|---:|
| pep_117 | 3.1773 | 2.6607 | 16.26% |
| pep_219 | 6.2636 | 6.2636 | 0.00% |
| pep_221 | 7.9537 | 4.0452 | 49.14% |
| pep_575 | 20.9322 | 15.1235 | 27.75% |
| pep_673 | 21.6679 | 21.6679 | 0.00% |
| pep_921 | 67.4329 | 11.0047 | 83.68% |
| pep_925 | 18.8208 | 16.0202 | 14.88% |
| pep_967 | 33.9040 | 33.9040 | 0.00% |
| pep_980 | 3.8768 | 2.5731 | 33.63% |
| pep_999 | 55.4269 | 29.5751 | 46.64% |

Seed 42; one trajectory per task. Original SemanticOPT generation and labeled random-fallback policy retained. Hardware changed to two A16 cards per replica; runtime is not directly comparable with the original hardware. FSBO uses seed 1. Offline FSBO/VAE training costs are separate.

See paired_results.csv and convergence.csv for measured values.
