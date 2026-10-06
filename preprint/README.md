# SafeDelta preprint

## A split-control criterion for few-shot adaptation in single-cell perturbation prediction

Wan-Li Liang, Ya-Bing Yang, Qiong-Yi Zhang, Jing-Cheng Feng, Qian-Fei Wang, Dehua Zou, Jia-Lin Tang, Yang Liu, Si-Yuan Wang, Junwei Duan, Yu Bai, Yi-Fang Li, Wan-Yang Sun, Yun-Feng Cao, Rong-Rong He

Wan-Li Liang, Ya-Bing Yang and Qiong-Yi Zhang contributed equally.
Correspondence: Wan-Yang Sun, Yun-Feng Cao and Rong-Rong He.

Author manuscript, 6 October 2026. This preprint has not been peer reviewed.
Submitted to bioRxiv; screening is in progress. The bioRxiv link and DOI will be
added once the preprint is posted.

- [Read the manuscript (PDF; main and Extended Data figures included)](SafeDelta_preprint.pdf)
- [Supplementary Note 1 (PDF)](SafeDelta_SupplementaryNote1.pdf)
- [Supplementary Table 1: control definitions (Excel)](SupplementaryTable1_control_definitions.xlsx)

## Abstract

Few-shot adaptation can help or harm single-cell perturbation prediction, but conventional evaluation shares a control estimate between calibration responses and held-out targets. We show how methods carry control error into their predictions and derive its contribution to score inflation. Nested control-depth experiments support inverse-depth scaling of inner-product inflation; correlation inflation also depends on the control noise-to-signal ratio. Independent-control scoring changed the rank of 40 of 52 configuration–dataset entries. We introduce SafeDelta, a split-control criterion that uses calibration data to decide whether to adapt. Across six datasets, mean gated gain was non-negative under the original two-control evaluation. On a deep plate, fixed predictions and decisions retained positive mean gains under three additional independent control references. The same reproducibility principle identified a fibrosis programme that localises to scar in human liver. These results inform adaptation decisions and control-depth planning.

## Citation

Liang, W.-L., Yang, Y.-B., Zhang, Q.-Y., et al. (2026).
A split-control criterion for few-shot adaptation in single-cell perturbation prediction. Author preprint.
https://github.com/yyb2020/safedelta/tree/main/preprint

## Rights

Copyright © 2026 the authors. The manuscript and supplementary files in this
directory are separate from the software and are not covered by its MIT licence.
The manuscript and supplementary materials are licensed under
[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
(CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/).
You may share these materials with attribution for noncommercial purposes;
you may not distribute modified versions. See the linked licence for its full terms.
