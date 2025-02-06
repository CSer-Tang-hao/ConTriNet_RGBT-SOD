# (TPAMI 2025) Divide-and-Conquer: Confluent Triple-Flow Network for RGB-T Salient Object Detection [![IEEE Page](https://img.shields.io/badge/IEEE-TPAMI.2024.3511621-green?style=flat-square)](https://ieeexplore.ieee.org/abstract/document/10778650) [![Arxiv Page](https://img.shields.io/badge/Arxiv-2412.01556-red?style=flat-square)](https://arxiv.org/abs/2412.01556)


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

## Framework
![framework](figs/framework.png)

> An overview of the proposed Confluent Triple-Flow Network (*ConTriNet*), which adopts an efficient **``Divide-and-Conquer''** strategy, is presented. *ConTriNet* comprises three main flows: a modality-complementary flow that predicts a modality-complementary saliency map, and two modality-specific flows that predict RGB- and Thermal-specific saliency maps, respectively.

## VT-IMAG Dataset 
![vt-imag](figs/VT-IMAG.png)

> The *primary purpose* of the constructed VT-IMAG is to drive the advancement of RGB-T SOD methods and facilitate their deployment in real-world scenarios. For a fair comparison, all models are solely trained on clear data and simple scenes (*i.e.*, training set of VT5000) and evaluated for **Zero-shot Robustness** on various real-world challenging cases in VT-IMAG. [Download Dataset (Google Drive)](https://drive.google.com/file/d/1xzvqoYLrmJ-6x33DygCP-LhFNYfhQL-u/view?usp=sharing)

## Benchmark Datasets

- [VT5000 (ArXiv)](https://arxiv.org/pdf/2007.03262.pdf) [Download Datasets (Google Drive)](https://drive.google.com/drive/folders/1So0dHK5-aKj1t6OmFhRGLh_0nsXbldZE?usp=sharing) 
- [VT1000 (ArXiv)](https://arxiv.org/pdf/1905.06741.pdf) [Download Datasets (Google Drive)](https://drive.google.com/drive/folders/1kEGOuljxKxIYwH54sNH_Wqmw7Sf7tTw5?usp=sharing) 
- [VT821 (ArXiv)](https://arxiv.org/pdf/1701.02829.pdf)  [Download Datasets (Google Drive)](https://drive.google.com/drive/folders/1gjTRVwvTNL0MJaJwS6vkpoi5rGyxIh41?usp=sharing)

## Saliency Maps

> The prediction results of existing RGB-T SOD methods and our ConTriNet on benchmark datasets are now available for download, enabling researchers to easily compare their methods with existing SOTA methods and directly incorporate these results into their studies.

- [Download VT5000 Saliency_maps (Google Drive)](https://drive.google.com/drive/folders/17sqNHH1NSyvDJgxW-1z65Ryn7p__zpV7?usp=sharing) 
- [Download VT1000 Saliency_maps (Google Drive)](https://drive.google.com/drive/folders/1ucKJxD6lzdJ1pKE3VR81ae9RHbdiXQBE?usp=sharing) 
- [Download VT821 Saliency_maps (Google Drive)](https://drive.google.com/drive/folders/1abbs3rcefsTSHFfBmPg8aFHxgCu78oIM?usp=sharing) 

> The prediction results of existing RGB-T SOD methods on VT-IMAG are now available for download, enabling researchers to easily compare their methods with existing SOTA methods and directly incorporate these results into their studies. 
- [Download VT-IMAG Saliency_maps (Google Drive)](https://drive.google.com/drive/folders/18YWuQ4R-uYLElQEBN3WykQPasUtrxOuj?usp=sharing)

## How to run
+ After you download datasets, just run `train.py` to train our model and `test.py` to generate the final prediction map.

```bash

python train.py --rgb_label_root [path_of_training_rgb_images] --thermal_label_root [path_of_training_thermal_images] --gt_label_root [path_of_training_gt_images] --gpu_id 
python test.py --test_path [path_of_test_images] --model_path [path_of_model_checkpoint] --gpu_id 

```
> The original model weights are unavailable due to the long paper review process. We reorganize and release the code along with newly reproduced [model weights (Google Drive)](https://drive.google.com/file/d/1YX-sxyZfhMOx6zCCGk8L9SLDHSlEV74X/view?usp=sharing) for reference, with results closely matching the paper, sometimes outperforming it.

## Evaluation

> We use this [Saliency-Evaluation-Toolbox](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox) for evaluating all RGB-T SOD results.

## Citation

Please cite our paper if you find the work useful, thanks!

    @article{tang2024divide,
      author={Tang, Hao and Li, Zechao and Zhang, Dong and He, Shengfeng and Tang, Jinhui},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
      title={Divide-and-Conquer: Confluent Triple-Flow Network for RGB-T Salient Object Detection}, 
      year={2025},
      volume={47},
      number={3},
      pages={1958-1974},
      doi={10.1109/TPAMI.2024.3511621}}




