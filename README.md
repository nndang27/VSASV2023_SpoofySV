# VSASV2023_SpoofySV

# 0x00 ASV
This directory provides code related to extracting speaker embeddings. 
We used some public pre-trained models and some models we trained ourselves on the Voxceleb2 dataset to extract speaker embeddings.
```bash
./run_spk_ebd_extract.sh
```
# 0x02 Test different ASV models and AASIST(CM) pre-train model

```bash
./run_test_nonorm.sh
```
```bash
./run_test_l2norm.sh
```

#  Result
## NoNorm Score Fusion

|  Model                                     | Public test EER(%) |
| ------------------------------------------ | :----------------- | 
| ECAPA-TDNN + ASSIST                        |       10.72        | 
| ECAPA-TDNN + S2pecNet                      |        8.86        | 


## L2Norm Score Fusion
|  Model                                     | Public test EER(%) | 
| ----------------------------------------   | :---------------   |
| ECAPA-TDNN + ASSIST                        |        4.68        |  
| ECAPA-TDNN + S2pecNet                      |        4.33        |  
| Res2Net    + S2pecNet                      |        3.65        |
| Res2Net + S2pecNet(perturb phase)          |        3.17        |
| hard voting (res2net + res2next + ecapa) + S2pecNet(perturb phase) |        2.86        |

