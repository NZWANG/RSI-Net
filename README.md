# RSI-Net
The implemented codes of RSI-Net are open here. 
If you used our RSI-Net codes, please cite our paper: He, S., Lu, X., Gu, J., Tang, H., Yu, Q., Liu, K., ... & Wang, N. (2022). RSI-Net: Two-Stream Deep Neural Network for Remote Sensing Images based Semantic Segmentation. IEEE Access.
If you have any question or collaboration suggestion about our method, please contact wangnizhuan1120@gmail.com. 

The codes of various networks were tested in Pytorch 1.6 version or higher versions(a little bit different from 0.8 version in some functions) in Python 3.8 on Ubuntu machines (May need minor changes on windows).

Regarding the graph encoding part, we refer to https://github.com/qichaoliu/CNN_Enhanced_GCN, and we are very grateful for their processing ideas.

The dataset is download on https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx. For more convenience, we directly crop the sliding window into 512×512 for processing.


