# STT Evaluation Report

- Clips: **8**
- Overall **WER 0.179** (95% CI 0.000-0.386, lower is better)
- Overall **CER 0.161**
- Errors: 3 sub, 0 ins, 12 del over 84 reference words
- Runaway / hallucination flags: **0** 
- Silence clips: **1**, hallucinated on **1** ['silence-001'] (1 invented words, not counted in WER)

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
| clean-001 | clean | 0.300 | 0.305 | 1/0/2 |  |
| clean-002 | clean | 0.000 | 0.000 | 0/0/0 |  |
| clean-003 | clean | 0.200 | 0.207 | 1/0/1 |  |
| noisy-001 | noisy | 0.000 | 0.000 | 0/0/0 |  |
| noisy-002 | noisy | 0.000 | 0.000 | 0/0/0 |  |
| accent-001 | accent | 0.000 | 0.000 | 0/0/0 |  |
| accent-002 | accent | 0.000 | 0.000 | 0/0/0 |  |
| numbers-001 | numbers | 0.714 | 0.746 | 1/0/9 |  |
