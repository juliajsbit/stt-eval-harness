# STT Evaluation Report

- Clips: **8**
- Overall **WER 0.179** (95% CI 0.000-0.386, lower is better)
- Overall **CER 0.189**
- Errors: 3 sub, 0 ins, 12 del over 84 reference words
- Runaway / hallucination flags: **0** 

## WER by slice

| Slice | WER |
| --- | --- |
| accent | 0.000 |
| clean | 0.167 |
| noisy | 0.000 |
| numbers | 0.714 |

## Per-clip

| ID | Slice | WER | CER | S/I/D | Runaway |
| --- | --- | --- | --- | --- | --- |
| clean-001 | clean | 0.300 | 0.322 | 1/0/2 |  |
| clean-002 | clean | 0.000 | 0.030 | 0/0/0 |  |
| clean-003 | clean | 0.200 | 0.241 | 1/0/1 |  |
| noisy-001 | noisy | 0.000 | 0.038 | 0/0/0 |  |
| noisy-002 | noisy | 0.000 | 0.051 | 0/0/0 |  |
| accent-001 | accent | 0.000 | 0.033 | 0/0/0 |  |
| accent-002 | accent | 0.000 | 0.034 | 0/0/0 |  |
| numbers-001 | numbers | 0.714 | 0.762 | 1/0/9 |  |
