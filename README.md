# CNLCU 

ICLR‘22: Sample Selection with Uncertainty of Losses for Learning with Noisy Labels (PyTorch implementation).  

This is the code for the paper:
[Sample Selection with Uncertainty of Losses for Learning with Noisy Labels](https://openreview.net/pdf?id=xENf4QUL4LW)      
Xiaobo Xia, Tongliang Liu, Bo Han, Mingming Gong, Jun Yu, Gang Niu, Masashi Sugiyama.


## Dependencies
We implement our methods by PyTorch on NVIDIA Tesla V100 GPU. The environment is as bellow:
- [Ubuntu 16.04 Desktop](https://ubuntu.com/download)
- [PyTorch](https://PyTorch.org/), version = 1.2.0
- [CUDA](https://developer.nvidia.com/cuda-downloads), version = 10.0
- [Anaconda3](https://www.anaconda.com/)

### Install requirements.txt
~~~
pip install -r requirements.txt
~~~

## Experiments
We verify the effectiveness of the proposed method on synthetic noisy datasets. In this repository, we provide the used [datasets](https://drive.google.com/open?id=1Tz3W3JVYv2nu-mdM6x33KSnRIY1B7ygQ) (the images and labels have been processed to .npy format). You should put the datasets in the folder “data” when you have downloaded them.       

If you find this code useful in your research, please cite  
```bash
@inproceedings{
  xia2022sample,
  title={Sample Selection with Uncertainty of Losses for Learning with Noisy Labels},
  author={Xiaobo Xia and Tongliang Liu and Bo Han and Mingming Gong and Jun Yu and Gang Niu and Masashi Sugiyama},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```  
