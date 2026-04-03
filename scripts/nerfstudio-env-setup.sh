#!/bin/bash

# How to add new commands:
# For any shell command:
# execute_command "Description" "command" "Error message"

# For conda packages:
# execute_conda_install "Description" "package names" "channel"

# For pip packages:
# execute_pip_install "Description" "package names" "extra args"

# =============================================================================
# Environment Setup Script with Comprehensive Error Handling
# =============================================================================

set -eo pipefail  # Exit on error and pipe failures (but allow unset variables for conda compatibility)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CONDA_ENV_NAME=nerfstudio
CONDA_ENV_PATH=/home/vamsik1211/Data/Projects/3D-Reconstructions/CityScapeGS/data/conda-envs/$CONDA_ENV_NAME
BASE_DIR=/home/vamsik1211/Data/Projects/3D-Reconstructions/CityScapeGS
# IMAGES_DIR="${BASE_DIR}/data/Urban3D-Dataset/ArtSci/Smith et al/ArtSci_coarse/Smith et al/ArtSci_coarse/images"
# WORK_DIR=/tmp/workdir
# TMPDIR=/tmp/tmp

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# Logging Functions
# -----------------------------------------------------------------------------
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_command() {
    echo -e "${CYAN}[COMMAND]${NC} Running: $1"
}

log_section() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

# -----------------------------------------------------------------------------
# Error Handling
# -----------------------------------------------------------------------------
error_exit() {
    log_error "$1"
    log_error "Script failed at line $2"
    exit 1
}

trap 'error_exit "An error occurred during execution" $LINENO' ERR

# -----------------------------------------------------------------------------
# Command Execution Framework
# -----------------------------------------------------------------------------
# Structure to easily add new commands:
# Each command has:
#   - description: What the command does
#   - cmd: The actual command to run
#   - error_msg: Custom error message if command fails

execute_command() {
    local description="$1"
    local cmd="$2"
    local error_msg="${3:-Command failed}"
    
    log_info "$description"
    log_command "$cmd"
    
    if eval "$cmd"; then
        log_success "✓ $description - Completed"
        return 0
    else
        local exit_code=$?
        log_error "$error_msg (Exit code: $exit_code)"
        return $exit_code
    fi
}

execute_conda_install() {
    local description="$1"
    local packages="$2"
    local channel="${3:-}"
    
    local cmd="conda install -y"
    if [ -n "$channel" ]; then
        cmd="$cmd -c $channel"
    fi
    cmd="$cmd $packages"
    
    execute_command "$description" "$cmd" "Failed to install: $packages"
}

execute_pip_install() {
    local description="$1"
    local packages="$2"
    local extra_args="${3:-}"
    
    local cmd="pip install $packages"
    if [ -n "$extra_args" ]; then
        cmd="$cmd $extra_args"
    fi
    
    execute_command "$description" "$cmd" "Failed to install Python package: $packages"
}

execute_cmake_build() {
    local description="$1"
    local source_dir="$2"
    local build_dir="$3"
    local cmake_args="$4"
    
    log_info "$description"
    
    execute_command "Creating build directory" "mkdir -p $build_dir" "Failed to create build directory"
    execute_command "Changing to build directory" "cd $build_dir" "Failed to change directory"
    execute_command "Running CMake configuration" "cmake $source_dir $cmake_args" "CMake configuration failed"
    execute_command "Building with Ninja" "ninja" "Build failed"
    execute_command "Installing" "ninja install" "Installation failed"
}

# -----------------------------------------------------------------------------
# Main Setup Commands
# -----------------------------------------------------------------------------

main() {
    log_section "Starting Environment Setup"
    log_info "Script started at $(date)"
    
    # -------------------------------------------------------------------------
    # Step 1: Conda Environment Creation
    # -------------------------------------------------------------------------
    log_section "Step 1: Creating Conda Environment"
    
    if [ -d "$CONDA_ENV_PATH" ]; then
        log_info "Conda environment already exists at $CONDA_ENV_PATH"
        log_success "✓ Using existing conda environment"
    else
        execute_command \
            "Creating conda environment '$CONDA_ENV_NAME' at $CONDA_ENV_PATH with Python 3.10" \
            "conda create -p $CONDA_ENV_PATH python=3.10 -y" \
            "Failed to create conda environment"
    fi
    
    # -------------------------------------------------------------------------
    # Step 2: Activate Environment
    # -------------------------------------------------------------------------
    log_section "Step 2: Activating Conda Environment"
    
    log_info "Sourcing conda initialization script"
    if ! source "$(conda info --base)/etc/profile.d/conda.sh"; then
        error_exit "Failed to source conda.sh" $LINENO
    fi
    log_success "✓ Conda initialization script sourced"
    
    log_info "Activating conda environment at $CONDA_ENV_PATH"
    if ! conda activate $CONDA_ENV_PATH; then
        error_exit "Failed to activate conda environment" $LINENO
    fi
    log_success "✓ Conda environment activated"
    
    # -------------------------------------------------------------------------
    # Step 3: Install System Packages
    # -------------------------------------------------------------------------
    log_section "Step 3: Installing System Packages via Conda"
    
    execute_conda_install "Installing multimedia and math libraries" "nvidia::cuda-toolkit==12.9.0 conda-forge::gcc conda-forge::gxx conda-forge::ceres-solver conda-forge::libcholmod"
    execute_conda_install "Installing Multimedia, Math, CUDA and GCC Libraries" \
        "ffmpeg eigen mkl cmake \
            conda-forge::freeimage conda-forge::libopenblas conda-forge::glew conda-forge::boost \
            conda-forge::glog conda-forge::gflags conda-forge::ninja conda-forge::qt conda-forge::cgal" \

    # execute_conda_install "Installing CUDA Toolkit 12.8.1" "" "nvidia/label/cuda-12.9.0"
    # execute_conda_install "Installing computer vision and optimization libraries" \
    #     "" \
    #     "conda-forge"
    
    # -------------------------------------------------------------------------
    # Step 4: Install PyTorch
    # -------------------------------------------------------------------------
    log_section "Step 4: Installing PyTorch with CUDA 12.8 Support"
    
    execute_pip_install \
        "Installing PyTorch 2.8.0, TorchVision 0.23.0, and TorchAudio 2.8.0 with CUDA 12.9" \
        "torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0" \
        "--index-url https://download.pytorch.org/whl/cu129"
    
    # -------------------------------------------------------------------------
    # Step 5: Setup Working Directories
    # -------------------------------------------------------------------------
    log_section "Step 5: Setting Up Working Directories"
    
    log_info "Setting up environment variables"
    export BASE_DIR
    export IMAGES_DIR
    export WORK_DIR
    # export TMPDIR
    # mkdir -p $TMPDIR
    log_success "✓ Environment variables exported"
    
    # execute_command \
    #     "Creating working directory at $WORK_DIR" \
    #     "mkdir -p $WORK_DIR" \
    #     "Failed to create working directory"
    
    # -------------------------------------------------------------------------
    # Step 6: Build and Install COLMAP
    # -------------------------------------------------------------------------
    log_section "Step 6: Building and Installing COLMAP"
    
    local colmap_src="$BASE_DIR/third_party/colmap"
    local colmap_build="$colmap_src/build"
    local colmap_cmake_args="-GNinja -DGUI_ENABLED=OFF -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DOPENGL_ENABLED=OFF -DGPU_ENABLED=OFF -DCUDA_ENABLED=OFF"
    
    if [ ! -d "$colmap_src" ]; then
        error_exit "COLMAP source directory not found at $colmap_src" $LINENO
    fi
    
    execute_command "Changing to COLMAP source directory" "cd $colmap_src" "Failed to access COLMAP directory"
    execute_cmake_build \
        "Building COLMAP from source" \
        ".." \
        "$colmap_build" \
        "$colmap_cmake_args"
    
    # -------------------------------------------------------------------------
    # Step 7: Install Hierarchical-Localization
    # -------------------------------------------------------------------------
    log_section "Step 7: Installing Hierarchical-Localization (hloc)"
    
    local hloc_dir="$BASE_DIR/third_party/Hierarchical-Localization"
    if [ ! -d "$hloc_dir" ]; then
        error_exit "Hierarchical-Localization directory not found at $hloc_dir" $LINENO
    fi
    
    execute_command "Changing to Hierarchical-Localization directory" "cd $hloc_dir" "Failed to access hloc directory"
    execute_pip_install "Installing Hierarchical-Localization in editable mode" "-e ."
    
    # -------------------------------------------------------------------------
    # Step 8: Install tiny-cuda-nn
    # -------------------------------------------------------------------------
    log_section "Step 8: Installing tiny-cuda-nn"
    
    execute_pip_install \
        "Installing tiny-cuda-nn from GitHub" \
        "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch" \
        "--no-build-isolation"
    
    # -------------------------------------------------------------------------
    # Step 9: Install gsplat
    # -------------------------------------------------------------------------
    log_section "Step 9: Installing gsplat with CUDA support"
    
    execute_pip_install \
        "Installing gsplat from GitHub with CUDA extensions" \
        "git+https://github.com/nerfstudio-project/gsplat.git" \
        "--no-build-isolation"
    
    # -------------------------------------------------------------------------
    # Step 10: Install Nerfstudio
    # -------------------------------------------------------------------------
    log_section "Step 10: Installing Nerfstudio"
    
    local nerfstudio_dir="$BASE_DIR/third_party/nerfstudio"
    if [ ! -d "$nerfstudio_dir" ]; then
        error_exit "Nerfstudio directory not found at $nerfstudio_dir" $LINENO
    fi
    
    execute_command "Changing to Nerfstudio directory" "cd $nerfstudio_dir" "Failed to access nerfstudio directory"
    execute_pip_install "Installing Nerfstudio in editable mode" "-e ."

    # -------------------------------------------------------------------------
    # Step 11: Install Pixel Perfect SFM
    # -------------------------------------------------------------------------
    # log_section "Step 11: Installing Pixel Perfect SFM"
    
    
    # -------------------------------------------------------------------------
    # Step 12: Process Data with Nerfstudio
    # -------------------------------------------------------------------------
    log_section "Step 12: Processing Image Data with Nerfstudio"
    
    # if [ ! -d "$IMAGES_DIR" ]; then
    #     log_warning "Images directory not found at $IMAGES_DIR"
    #     log_warning "Skipping data processing step"
    # else
    #     local ns_cmd="ns-process-data images \
    #         --data '$IMAGES_DIR' \
    #         --output-dir '$WORK_DIR/nerf_data' \
    #         --camera-type perspective \
    #         --matching-method sequential \
    #         --sfm-tool hloc \
    #         --feature-type superpoint_max \
    #         --matcher-type 'superpoint+lightglue' \
    #         --num-downscales 0 \
    #         --use-single-camera-mode"
        
    #     execute_command \
    #         "Processing images with Nerfstudio pipeline" \
    #         "$ns_cmd" \
    #         "Failed to process image data"
    # fi

    # -------------------------------------------------------------------------
    # Step 13: Run the ns-train Command
    # -------------------------------------------------------------------------
    log_section "Step 13: Training with Nerfstudio"

    # local ns_cmd="ns-train splatfacto \
    #     --data '$WORK_DIR/nerf_data' \
    #     --output-dir '$WORK_DIR/nerf_output' \
    #     --max-num-iterations 30000 \
    #     --viewer.make-share-url True \
    #     --experiment-name 'art_sci_instant_ngp'"

    # if [ ! -d "$WORK_DIR/nerf_data" ]; then
    #     log_warning "Nerf data directory not found at $WORK_DIR/nerf_data"
    #     log_warning "Skipping training step"
    # else
    #     execute_command \
    #         "Training Nerfstudio model" \
    #         "$ns_cmd" \
    #         "Failed to train Nerfstudio model"
    # fi

    # -------------------------------------------------------------------------
    # Completion
    # -------------------------------------------------------------------------
    log_section "Setup Complete!"
    log_success "All installation and setup steps completed successfully!"
    log_info "Script finished at $(date)"
    log_info "Conda environment: $CONDA_ENV_PATH"
    log_info "Work directory: $WORK_DIR"
    log_info ""
    log_info "To activate the environment in the future, run:"
    echo "  conda activate $CONDA_ENV_PATH"
}

# -----------------------------------------------------------------------------
# Execute Main Function
# -----------------------------------------------------------------------------
main "$@"
