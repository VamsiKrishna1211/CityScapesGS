import argparse
import sys
import os

# Ensure custom_gaussiansplat is in the Python path if run from outside
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import GaussianModel

def main():
    parser = argparse.ArgumentParser(description="Convert custom GS checkpoint to original 3DGS format.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input custom GS checkpoint (.pth)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output original GS checkpoint (.pth)")
    parser.add_argument("--iteration", type=int, default=0, help="Iteration number to embed in the output checkpoint")
    parser.add_argument("--spatial_lr_scale", type=float, default=1.0, help="Spatial LR scale used by original GS")
    
    args = parser.parse_args()
    
    print(f"Loading custom GS model from: {args.input}")
    try:
        model = GaussianModel.from_checkpoint(args.input)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
        
    print(f"Exporting to original GS format: {args.output}")
    try:
        model.export_to_original_gs_checkpoint(
            path=args.output,
            iteration=args.iteration,
            spatial_lr_scale=args.spatial_lr_scale
        )
        print("Export complete.")
    except Exception as e:
        print(f"Failed to export model: {e}")

if __name__ == "__main__":
    main()
