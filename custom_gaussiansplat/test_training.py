"""
Example script to test the custom Gaussian Splatting implementation
"""
import torch
from train import train_pipeline
from model import GaussianModel

def test_basic_training():
    """Test basic training pipeline with a small dataset"""
    
    # Example paths - adjust to your data
    colmap_path = 'data/boston_colmap/sparse/0'
    images_path = 'data/boston/images'
    output_dir = './outputs/test_run'
    
    print("=" * 60)
    print("Testing Custom Gaussian Splatting Implementation")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Training will be very slow!")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\nCOLMAP path: {colmap_path}")
    print(f"Images path: {images_path}")
    print(f"Output dir: {output_dir}")
    print()
    
    # Run training
    try:
        model = train_pipeline(
            colmap_path=colmap_path,
            images_path=images_path,
            output_dir=output_dir
        )
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        
        # Save PLY file
        ply_path = f'{output_dir}/final_gaussians.ply'
        model.save_ply(ply_path)
        print(f"\nExported to: {ply_path}")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False

def test_checkpoint_loading():
    """Test loading a saved checkpoint"""
    checkpoint_path = './outputs/test_run/model_final.pt'
    
    print("\n" + "=" * 60)
    print("Testing Checkpoint Loading")
    print("=" * 60)
    
    try:
        model = GaussianModel.load_checkpoint(checkpoint_path, device='cuda')
        print(f"Successfully loaded model with {len(model._means)} Gaussians")
        return True
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        return False

if __name__ == '__main__':
    import sys
    import os
    
    # Check if paths exist
    if not os.path.exists('data/boston_colmap/sparse/0'):
        print("ERROR: COLMAP data not found at 'data/boston_colmap/sparse/0'")
        print("Please update paths in this script to match your data location")
        sys.exit(1)
    
    # Test training
    success = test_basic_training()
    
    # Test checkpoint loading if training succeeded
    if success:
        test_checkpoint_loading()
    
    print("\n" + "=" * 60)
    print("All tests completed")
    print("=" * 60)
