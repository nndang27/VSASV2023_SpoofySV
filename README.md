# VSASV2023 SpoofySV team

This repository provides our solution for the challenge VSASV in VLSP 2023.
You can check the final result in ['VSASV2023 submission'](https://aihub.ml/competitions/602#results)

Baseline models:
ASV task in ['ECAPA-TDNN'](https://github.com/TaoRuijie/ECAPA-TDNN)
CM task in ['VSASV2023 submission'](https://github.com/clovaai/aasist)

Fine-tune the newest state of the art CM model that published in INTERSPEECH 2023 ['S2pecNet'](https://github.com/ph-w2000/S2pecNet)
Then we develop CM model by technique to perturb phase. Read in ['Phase perturbation improves channel robustness for speech spoofing countermeasures'](https://arxiv.org/abs/2306.03389)

To generate more data for training, we apply voice conversion ['VQ-VAE'](https://github.com/bshall/ZeroSpeech) and voice cloning ['Vietnamese Voice Cloning'](https://github.com/v-nhandt21/ViSV2TTS ) 

For cleaning training set, we extract the embedding of each audio by a self-supervised ASV model ['Loss-Gated Learning'](https://github.com/TaoRuijie/Loss-Gated-Learning?fbclid=IwAR1q4MSfjWU8y5FeMm3X07zyzB3JmFaH52gNPFB6QMFYUZ_5ggstKHThovE)
Then we use DBSCAN clustering to distinguish multiple voices, noise, music in an id and only keep the main voice.

# Test different ASV and CM models

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

