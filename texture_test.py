import numpy as np
import torch
import cv2
import os
from lib.utils.config import get_cfg_defaults
from lib.utils.uv_sample.divided_uv_generator import Index_UV_Generator
from lib.tools.texture_projection import texture_projection
import yaml

def main():
    # Configuration and initialization
    cfg = get_cfg_defaults("./examples/demo_image/config.yaml")
    
    # Load resources configuration
    with open('lib/configs/resources.yaml', 'r') as f:
        resources_dict = yaml.safe_load(f)
    
    cfg_resources = type('ConfigResources', (), {
        'masks': type('Masks', (), {'mask_wo_eyes': resources_dict['masks']['mask_wo_eyes']})(),
        'econ_calib': resources_dict['econ_calib'],
        'models': type('Models', (), {'smplx_vts_template': resources_dict['models']['smplx_vts_template']})()
    })
    
    # Verify input files exist
    files = {
        'completed_mesh': './examples/demo_image/demo_image_smpl_00.obj',
        'image': './examples/demo_image/demo_image.png',
        'mask': './examples/demo_image/demo_image_mask.png'
    }
    
    for key, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find {key} file at {path}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize UV sampler
    uv_sampler = Index_UV_Generator(
        UV_height=1024,
        UV_width=1024,
        uv_type='SMPLX',
        data_dir='data/smplx'
    ).to(device)  # Move UV sampler to device
    
    save_root = 'output/texture'
    os.makedirs(save_root, exist_ok=True)
    
    try:
        tex_path, mask_path, mesh_path = texture_projection(
            files=files,
            cfg=cfg,
            cfg_resources=cfg_resources,
            uv_sampler=uv_sampler,
            device=device,
            save_root=save_root
        )
        print("Texture projection completed successfully!")
        print(f"Results saved to:\n{tex_path}\n{mask_path}\n{mesh_path}")
    except Exception as e:
        print(f"Error during texture projection: {str(e)}")

if __name__ == "__main__":
    main()