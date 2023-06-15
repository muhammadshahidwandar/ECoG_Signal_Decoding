# Hand Trajectory prediction from ECoG signal using statistical methods (Partial Least Square Regression(PLS) and Higher Order PLS)

*PLS and HOPLS Code is taken from this repository(https://github.com/arthurdehgan/HOPLS)

## Overview
![BlockDiagram](https://github.com/muhammadshahidwandar/ECoG_Signal_Decoding/blob/main/images/Ecog_Signal_Decode.jpg)

## Results
![Results](https://github.com/muhammadshahidwandar/ECoG_Signal_Decoding/blob/main/images/Hand_trajactory.jpeg)

## Sub-directories and Files
There are two sub-directories described as follows::

### images
Containes over all block diagram and visual results of predicted signal.
### source
Contains source code for data read functions, feature extraction using scalogram, and application of statistical regression methods for predicting 3D hand position trajectory.


## Dependencies
* python 3.7
* tensorly 0.7.0
* torch 1.13.0 
* scipy  1.9.3


## Dataset
The data is downloaded from the link (http://neurotycho.org/), which is referenced in the paper.



## Reference

**Long-term asynchronous decoding of arm motion using electrocorticographic signals in monkey**  
```
@article{chao2010long,
  title={Long-term asynchronous decoding of arm motion using electrocorticographic signals in monkey},
  author={Chao, Zenas C and Nagasaka, Yasuo and Fujii, Naotaka},
  journal={Frontiers in neuroengineering},
  pages={3},
  year={2010},
  publisher={Frontiers}
}
