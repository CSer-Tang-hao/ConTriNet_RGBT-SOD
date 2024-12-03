# Divide-and-Conquer: Confluent Triple-Flow Network for RGB-T Salient Object Detection [![Arxiv Page](https://img.shields.io/badge/Arxiv-2412.01556-red?style=flat-square)](https://arxiv.org/abs/2412.01556)


<div align="center">
    <a href='https://scholar.google.com/citations?hl=zh-CNJ' target='_blank'>Hao Tang<sup>1</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=L6J2V3sAAAAJ&hl=zh-CN' target='_blank'>Zechao Li<sup>1</sup></a>&emsp; 
    <a href='https://scholar.google.com.hk/citations?hl=zh-CN&user=zxVy7sIAAAAJ' target='_blank'>Dong Zhang<sup>2</sup></a>&emsp; 
    <a href='https://scholar.google.com/citations?user=rBWnK8wAAAAJ&hl=en' target='_blank'>Shengfeng He<sup>3</sup></a>&emsp; 
    <a href='https://scholar.google.com/citations?user=ByBLlEwAAAAJ&hl=zh-CN' target='_blank'>Jinhui Tang<sup>1</sup></a> 
</div> 

<div align="center">
    <sup>1</sup>Nanjing University of Science and Technology, Nanjing, China</br>
    <sup>2</sup>Hong Kong University of Science and Technology, Hong Kong, China</br>
    <sup>3</sup>Singapore Management University, Singapore&emsp;</br>
    
</div>

 -----------------

> Codes, Datasets, and Results Coming Soon!

## Framework
![framework](figs/framework.png)

## VT-IMAG Dataset 
![vt-imag](figs/VT-IMAG.png)

> The **primary purpose** of the constructed VT-IMAG is to drive the advancement of RGB-T SOD methods and facilitate their deployment in real-world scenarios. For a fair comparison, all models are solely trained on clear data and simple scenes (*i.e.*, training set of VT5000) and evaluated for **Zero-shot Robustness** on various real-world challenging cases in VT-IMAG. [Download Dataset (Google Drive)](https://drive.google.com/file/d/1xzvqoYLrmJ-6x33DygCP-LhFNYfhQL-u/view?usp=sharing)

> The prediction results of existing RGB-T SOD methods on VT-IMAG are now available for download, enabling researchers to easily compare their methods with existing SOTA methods and directly incorporate these results into their studies. [Download Saliency_maps (Google Drive)]() 

## Requirements

 - `Python 3.6`
 - [`Pytorch`](http://pytorch.org/) >= 1.7.0 
 - `Torchvision` = 0.10

## Evaluation

> We use this [Saliency-Evaluation-Toolbox](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox) for evaluating all RGB-T SOD results.

## Citation

Please cite our paper if you find the work useful, thanks!

    @ARTICLE{10113165,
       author={Tang, Hao and Li, Zechao and Zhang, Dong and He, Shengfeng and Tang, Jinhui},
       journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
       title={Divide-and-Conquer: Confluent Triple-Flow Network for RGB-T Salient Object Detection}, 
       year={2024},
       doi={}
    }

**[⬆ back to top](#1-preface)**

