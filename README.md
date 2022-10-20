MBW: Multiview Bootstrapping in the Wild (NeurIPS 2022)
============



<img src="https://upload.wikimedia.org/wikipedia/en/thumb/0/08/Logo_for_Conference_on_Neural_Information_Processing_Systems.svg/1200px-Logo_for_Conference_on_Neural_Information_Processing_Systems.svg.png" width=200>

### [Paper](https://arxiv.org/abs/2210.01721) | [Project page](https://multiview-bootstrapping-in-wild.github.io) <br>

### [MBW-Zoo dataset github](https://github.com/mosamdabhi/MBW-Data) [[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7058567.svg)]](https://doi.org/10.5281/zenodo.7058567) | [MBW pretrained models: ![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7054596.svg)](https://doi.org/10.5281/zenodo.7054596)  <br>

 




Multiview Bootstrapping in the Wild (MBW) provides a powerful way of generating labeled data in the wild at scale. Thereby, it democratizes the domain of data collection to a plethora of machine learning-driven computer vision applications via its novel self-supervision technique. 

<p align="center">
  <img width="900" src=graphics/overview.gif>
</p>

&nbsp;

Requirements (if using GPU)
============
- Tested in ``Pytorch 1.11``, with ``CUDA 11.4``

&nbsp;

Setup
============
1. Create a conda environment and activate it.
    ```
    conda env create -f environment_<cpu/gpu>.yml (change the flag within <> based on the available system)
    conda activate mbw
    pip install opencv-python
    ```
2. Please do a clean install of the submodule [`robust_loss_pytorch`](https://github.com/jonbarron/robust_loss_pytorch):
    ```
    cd modules/helpers/robust_loss_pytorch
    pip install git+https://github.com/jonbarron/robust_loss_pytorch
    ```
3. Please do a clean install of the submodule [`torch_batch_svd`](https://github.com/KinglittleQ/torch-batch-svd): (if using GPU)
    ```
    cd modules/helpers/torch-batch-svd
    export CUDA_HOME=/your/cuda/home/directory/    
    python setup.py install
    ```

&nbsp;

Data & Pre-trained models
============

1. Fetch the pre-trained flow and detector models from Zenodo using:

    ``` 
    zenodo_get 10.5281/zenodo.7054596
    unzip models.zip
    rm -rf models.zip && rm -rf md5sums.txt
    ```

2. Download the data from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7058567.svg)](https://doi.org/10.5281/zenodo.7058567) and unzip it in the `data` directory. 

   ```
   zenodo_get 10.5281/zenodo.7058567
   unzip data.zip
   rm -rf data.zip && rm -rf md5sums.txt
   ```
   The final directory after retrieving pre-trained models and sample data should look like this:
   ```
   ${mbw}
    `-- data
        `-- Chimpanzee
            |-- annot/
            |-- images/
        
    `-- models
        |-- detector/
        |-- flow/
        |-- mvnrsfm/
    ```
  
      
         
&nbsp;


Run unit tests
============

    ./scripts/unit_tests.sh


Training (Generate labels from MBW)
============

    ./scripts/train.sh

Evaluation and visualization
============
    ./scripts/eval.sh
    ./scripts/visualize.sh    


Jupyter notebook example
============
<img width="1111" alt="comingsoon" src="https://user-images.githubusercontent.com/6929121/87441911-486bf600-c611-11ea-9d45-94c215733cf7.png">


### Citation
If you use our code, dataset, or models in your research, please cite with:
```

@inproceedings{dabhi2022mbw,
	title={MBW: Multi-view Bootstrapping in the Wild},
	author={Dabhi, Mosam and Wang, Chaoyang and Clifford, Tim and Jeni, Laszlo and Fasel, Ian and Lucey, Simon},
	booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
	year={2022},
	ee = {https://openreview.net/forum?id=i1bFPSw42W0},
	organization={NeurIPS}
}


@inproceedings{dabhi2021mvnrsfm,
	title={High Fidelity 3D Reconstructions with Limited Physical Views},
	author={Dabhi, Mosam and Wang, Chaoyang and Saluja, Kunal and Jeni, Laszlo and Fasel, Ian and Lucey, Simon},
	booktitle={2021 International Conference on 3D Vision (3DV)},
	year={2021},
	ee = {https://ieeexplore.ieee.org/abstract/document/9665845},
	organization={IEEE}
}

```

