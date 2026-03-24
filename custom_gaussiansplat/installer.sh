#!/usr/bin/env bash
set -euo pipefail

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-citysplat}"

if ! command -v conda >/dev/null 2>&1; then
	echo "Error: conda command not found. Please install Conda and ensure it is on PATH."
	exit 1
fi

eval "$(conda shell.bash hook)"

conda create -n "$CONDA_ENV_NAME" python=3.12 -y
conda activate "$CONDA_ENV_NAME"

conda install -y -c conda-forge "libstdcxx-ng>=13.1.0" "libgcc-ng>=13.1.0" openssl ffmpeg
conda install -y -c conda-forge gcc_linux-64 gxx_linux-64

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

conda install -c nvidia -c conda-forge cuda-toolkit=12.8 -y
conda install -c 
export CUDA_HOME="$CONDA_PREFIX"
conda update -c conda-forge gcc_linux-64 gxx_linux-64 -y

python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install "git+https://github.com/nerfstudio-project/gsplat.git#v1.4.0" --no-build-isolation
python -m pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation

python -m pip install pycolmap tensorboard torchmetrics rich matplotlib splines nerfview opencv-python psutil nvidia-ml-py3 plyfile
python -m pip install lpips
python -m pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation

python -m pip install "git+https://gitlab.inria.fr/bkerbl/simple-knn.git" --no-build-isolation --force-reinstall --no-cache-dir

conda install conda-forge::colmap -y
