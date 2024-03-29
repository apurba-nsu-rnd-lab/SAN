<h1 align="center">
  <br>
    Rethinking Task-Incremental Learning Baselines
  <br>
</h1>

<h4 align="center">Official imlpementation of the paper [ICPR 2022]</h4>
<h3 align="left">
    Link to the Paper - <a href="https://arxiv.org/abs/2205.11367">arXiv </a> 
</h4>

### Prerequisites:
- Linux-64
- Python 3.9
- PyTorch 1.10.1
- CPU or NVIDIA GPU + CUDA10.2 CuDNN7.5

### Installation
- Create a conda environment and install required packages:
```bash
conda create -n <env> python=3.9
conda activate <env>
pip install -r requirements.txt
```

### Datasets
Download the *Mini-imagenet* and *notMNIST* datasets from [Google Drive](https://drive.google.com/drive/folders/1qRgXuuX8fvoSiAGSwn6UZYUdl65TVA24?usp=sharing). Other datasets will be automatically downloaded.

### Training
For training, run the following command.  
` python run.py -exp <experiment_id>`

To manually input number of runs, epochs and learning rate, run the following command:   
` python run.py -exp <experiment_id> -r <n_runs> -e <n_epochs> -lr <learning_rate> `

### Test
For test, run the following command.   
` python test.py -exp <experiment_id>`

## Citation

If you wish to cite this project, feel free to use this [BibTeX](http://www.bibtex.org/) reference:

```bibtex
@misc{https://doi.org/10.48550/arxiv.2205.11367,
  doi = {10.48550/ARXIV.2205.11367},
  url = {https://arxiv.org/abs/2205.11367},
  author = {Hossain, Md Sazzad and Saha, Pritom and Chowdhury, Townim Faisal and Rahman, Shafin and Rahman, Fuad and Mohammed, Nabeel},
  keywords = {Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Rethinking Task-Incremental Learning Baselines},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
