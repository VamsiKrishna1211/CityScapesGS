export CONDA_ENV_NAME=custom_gaussiansplat

conda create -n custom_gaussiansplat python=3.12 -y

conda activate custom_gaussiansplat

conda install -c nvidia cuda-toolkit=12.8 -y
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install "git+https://github.com/nerfstudio-project/gsplat.git#v1.4.0" --no-build-isolation

pip install pycolmap tensorboard torchmetrics rich matplotlib splines nerfview 
conda install -y -c conda-forge "libstdcxx-ng>=13.1.0" "libgcc-ng>=13.1.0"

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc  
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

pip install "git+https://gitlab.inria.fr/bkerbl/simple-knn.git" --no-build-isolation --force-reinstall --no-cache-dir

conda install conda-forge::colmap
