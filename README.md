# RSI-Net
The implemented codes of RSI-Net are open here. 
If you used our RSI-Net codes, please cite our paper: He, S., Lu, X., Gu, J., Tang, H., Yu, Q., Liu, K., ... & Wang, N. (2022). RSI-Net: Two-Stream Deep Neural Network for Remote Sensing Images-Based Semantic Segmentation. IEEE Access, 10, 34858-34871. https://ieeexplore.ieee.org/abstract/document/9745103/
If you have any question or collaboration suggestion about our method, please contact wangnizhuan1120@gmail.com. 

The codes of various networks were tested in Pytorch 1.6 version or higher versions(a little bit different from 0.8 version in some functions) in Python 3.8 on Ubuntu machines (May need minor changes on windows).

Regarding the GCN encoder part, we refer to https://github.com/qichaoliu/CNN_Enhanced_GCN, and we are very grateful to their processing ideas.

The example dataset can be downloaded from https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx. For more convenience, we directly crop the sliding window into 512×512 for processing.


