# EmotionMIL: An End-to-End Multiple Instance Learning Framework for Emotion Recognition from EEG Signals

## Overview

![Framework](https://github.com/yuty2009/emotionmil/blob/main/figures/framework.png)
**The EmotionMIL framework for emotion recognition from multi-channel EEG signals**. (a) EEG signal segmentation and preprocessing. (b) Temporal mixer layer for capturing temporal dependencies within EEG segments. (c) Spatial mixer layer for capturing spatial dependencies between EEG channels. (d) EEGMixer for instance feature extraction. (e) Multiple instance pooling layer for aggregating instance features and predicting the overall emotion label. (f) Detailed Rettention-based MIL pooling layer.

## Prepare the data

Download the datasets (access requirements may apply) from the following links:

- [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)

Replace the data path in the code with the path to the downloaded data. Run the "xxxreader.py" to preprocess the data. For example, to preprocess the DEAP dataset, run the following code:

```python
python deapreader.py
```

## Run the code

### Subject dependent cross-validation

To run the code for subject-dependent cross-validation, use the following command:

```python
python main_intrasub.py --dataset deap --arch eegmixer --mil retmil
```

### Subject independent leave-one-out cross-validation

To run the code for subject-independent leave-one-out cross-validation, use the following command:

```python
python main_crosssub.py --dataset deap --arch eegmixer --mil retmil
```

## Citation

If you use the code or results in your research, please consider citing our work at:

```
@article{yu2025emotionmil,
  title={EmotionMIL: An End-to-End Multiple Instance Learning Framework for Emotion Recognition from EEG Signals},
  author={Jun Xiao, Feifei Qi, Lingli Wang, Yanbin He, Jingang Yu, Wei Wu, Zhuliang Yu, Yuanqing Li, Zhenghui Gu, Tianyou Yu},
  journal={IEEE Transactions on Affective Computing},
  year={2025},
  pages={1-17},
  doi={10.1109/TAFFC.2025.3581388},
  url={https://ieeexplore.ieee.org/abstract/document/11045084},
}
```