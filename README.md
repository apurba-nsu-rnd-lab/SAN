# SAN-Draft

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

### Training
For training, run the following command.  
` python run.py `

To manually input number of runs, epochs and learning rate, run the following command:   
` python run.py -exp <experiment_id> -r <n_runs> -e <n_epochs> -lr <learning_rate> `

### Test
For test, run the following command.   
` python test.py `
