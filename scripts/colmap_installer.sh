#!/bin/bash

# Exit on any error
set -e

# This code assumes cuda is installed and set to CUDA_HOME

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${SCRIPT_DIR}/colmap_build"

echo "Installation directory: ${INSTALL_DIR}"

# Create and move to installation directory
mkdir -p "${INSTALL_DIR}"
cd "${INSTALL_DIR}"

export NUM_PARALLEL_JOBS=$(nproc)
export CUDA_ARCHITECTURES="70;75;80;86;89"
export SUITESPARSE_CUDA_ARCHITECTURES="70;75;80;86;89"
export SUITESPARSE_VERSION=v7.11.0
export CUDSS_CMAKE_DIR="${INSTALL_DIR}/cudss/current/lib/cmake/cudss"
export COLMAP_GIT_COMMIT=3.11.1

# Error handler function
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# Detect OS and install dependencies if Ubuntu/Debian
echo "========================================="
echo "Detecting operating system..."
echo "========================================="
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    echo "Detected OS: $OS"
    
    if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        echo "========================================="
        echo "Installing dependencies for $OS..."
        echo "========================================="
        
        # Check if running as root
        if [ "$EUID" -ne 0 ]; then
            echo "Installing dependencies requires root privileges. Please run with sudo or as root."
            echo "Attempting to use sudo..."
            SUDO_CMD="sudo"
        else
            SUDO_CMD=""
        fi
        
        $SUDO_CMD apt-get update || error_exit "Failed to update apt package list"
        $SUDO_CMD apt-get install -y --no-install-recommends \
            git ninja-build build-essential ccache \
            libboost-program-options-dev libboost-graph-dev libboost-system-dev \
            libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev libatlas-base-dev \
            libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev \
            libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libc6-dev \
            screen tmux htop unzip pkg-config \
            wget bzip2 curl ca-certificates || error_exit "Failed to install dependencies"
        
        echo "Dependencies installed successfully!"
    else
        echo "OS is not Ubuntu or Debian. Skipping dependency installation."
        echo "Please ensure required dependencies are installed manually."
    fi
else
    echo "Warning: Could not detect OS. /etc/os-release not found."
    echo "Please ensure required dependencies are installed manually."
fi

echo "========================================="
echo "Building SuiteSparse..."
echo "========================================="
if [ ! -d "suitesparse" ]; then
    git clone --recursive --depth=1 --branch ${SUITESPARSE_VERSION} \
        https://github.com/DrTimothyAldenDavis/SuiteSparse.git suitesparse || error_exit "Failed to clone SuiteSparse"
fi

cd suitesparse || error_exit "Failed to enter suitesparse directory"

# Build with the CUDA 12 toolchain from conda
cmake -S . -B build \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DSUITESPARSE_ENABLE_PROJECTS="suitesparse_config;amd;btf;camd;ccolamd;colamd;cholmod;spqr" \
    -DSUITESPARSE_USE_CUDA=ON \
    -DCHOLMOD_USE_CUDA=ON \
    -DSPQR_USE_CUDA=ON \
    -DSUITESPARSE_CUDA_ARCHITECTURES="${SUITESPARSE_CUDA_ARCHITECTURES}" \
    -DCUDAToolkit_ROOT=${CONDA_PREFIX} \
    -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} \
    -DCMAKE_CUDA_COMPILER=${CUDACXX} \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_INSTALL_RPATH="${CONDA_PREFIX}/lib" || error_exit "Failed to configure SuiteSparse"

cmake --build build -j ${NUM_PARALLEL_JOBS} || error_exit "Failed to build SuiteSparse"
cmake --install build || error_exit "Failed to install SuiteSparse"

cd "${INSTALL_DIR}" || error_exit "Failed to return to install directory"

echo "========================================="
echo "Downloading and setting up CUDSS..."
echo "========================================="
if [ ! -d "cudss" ]; then
    mkdir cudss || error_exit "Failed to create cudss directory"
fi

cd cudss || error_exit "Failed to enter cudss directory"

if [ ! -f "libcudss-linux-x86_64-0.6.0.5_cuda12-archive.tar.xz" ]; then
    wget https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-0.6.0.5_cuda12-archive.tar.xz || error_exit "Failed to download CUDSS"
fi

if [ ! -d "libcudss-linux-x86_64-0.6.0.5_cuda12-archive" ]; then
    tar -xf libcudss-linux-x86_64-0.6.0.5_cuda12-archive.tar.xz || error_exit "Failed to extract CUDSS"
fi

if [ ! -L "current" ]; then
    ln -s libcudss-linux-x86_64-0.6.0.5_cuda12-archive current || error_exit "Failed to create CUDSS symlink"
fi

cd "${INSTALL_DIR}" || error_exit "Failed to return to install directory"

echo "========================================="
echo "Building Ceres Solver..."
echo "========================================="
if [ ! -d "ceres" ]; then
    git clone --recurse-submodules https://github.com/ceres-solver/ceres-solver ceres || error_exit "Failed to clone Ceres"
fi

cd ceres || error_exit "Failed to enter ceres directory"
git checkout 8c50a34a1cac220ab2e7e2093b35b0db7e2a6e9b || error_exit "Failed to checkout Ceres commit"

cmake -S . -B build \
    -GNinja \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TESTING=OFF \
    -DUSE_CUDA=ON \
    -DCERES_NO_CUDA=OFF \
    -Dcudss_DIR=${CUDSS_CMAKE_DIR} \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_INSTALL_RPATH="${INSTALL_DIR}/cudss/current/lib" || error_exit "Failed to configure Ceres"

cmake --build build -j ${NUM_PARALLEL_JOBS} || error_exit "Failed to build Ceres"
cmake --install build || error_exit "Failed to install Ceres"

cd "${INSTALL_DIR}" || error_exit "Failed to return to install directory"

echo "========================================="
echo "Building COLMAP..."
echo "========================================="
if [ ! -d "src" ]; then
    git clone --depth=1 --branch=${COLMAP_GIT_COMMIT} https://github.com/colmap/colmap.git src || error_exit "Failed to clone COLMAP"
fi

cd src || error_exit "Failed to enter src directory"

cmake -S . -B build \
    -GNinja \
    -DGUI_ENABLED=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -Dcudss_DIR=${CUDSS_CMAKE_DIR} \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_INSTALL_RPATH="${INSTALL_DIR}/cudss/current/lib" \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_SYSTEM_LIBRARY_PATH="/lib/x86_64-linux-gnu;/usr/lib/x86_64-linux-gnu;/usr/local/lib" \
    -DCMAKE_LIBRARY_PATH="/lib/x86_64-linux-gnu;/usr/lib/x86_64-linux-gnu;/usr/local/lib" || error_exit "Failed to configure COLMAP"

cmake --build build -j ${NUM_PARALLEL_JOBS} || error_exit "Failed to build COLMAP"
cmake --install build || error_exit "Failed to install COLMAP"

cd "${INSTALL_DIR}" || error_exit "Failed to return to install directory"

echo "========================================="
echo "Installation completed successfully!"
echo "========================================="
echo "All files installed in: ${INSTALL_DIR}"
