conda env create --file setup/conda_env_cuda.yaml
conda activate TimeSeAD
pip install -e . 
pip install -e .[experiments] 
pip install sympy
