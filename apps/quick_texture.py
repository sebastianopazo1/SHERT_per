import numpy as np
import logging
import torch
import argparse
import os
import cv2
from lib.utils.common_util import check_key, check_files
from lib.utils.config import get_cfg_defaults
from lib.utils.uv_sample.divided_uv_generator import Index_UV_Generator
from lib.tools.texture_projection import texture_projection

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0, help='The GPU device to be used.')
    parser.add_argument('-i', '--input', type=str, default=None, help='The root path used for files loading.')
    parser.add_argument('-o', '--output', type=str, default=None, help='The folder used for output.')
    parser.add_argument('-r', '--resources', type=str, default='./lib/configs/resources.yaml', 
                       help='The resources file path.')
    return parser.parse_args()

def load_config(args):
    if args.input is not None:
        cfg_path = os.path.join(args.input, 'config.yaml')
        if not os.path.exists(cfg_path):
            raise Exception(f'Cannot find config in given root \'{args.input}\'.')
    else:
        cfg_path = './examples/demo_image/config.yaml'

    cfg = get_cfg_defaults(cfg_path)
    logging.info(f'Load configs from \'{cfg_path}\'.')
    return cfg, cfg_path

def process_texture(args, cfg, cfg_resources, device):
    # Initialize UV sampler with error handling
    try:
        uv_sampler = Index_UV_Generator(
            data_dir=cfg_resources.others.smplx_official_template_root,
            UV_height=1024,  # Explicit size parameters
            UV_width=1024
        ).to(device)
    except Exception as e:
        logging.error(f"Failed to initialize UV sampler: {str(e)}")
        raise

    # Define root directory
    root = args.input if args.input is not None else cfg.root
    logging.info(f'Load files from \'{root}\'.')

    # Define output directory
    save_root = args.output if args.output is not None else os.path.join(root, 'results')
    os.makedirs(save_root, exist_ok=True)
    logging.info(f'Save results to \'{save_root}\'.')

    # Load required files
    files = {}
    required_keys = ['image', 'mask', 'mesh']
    
    if check_key(cfg, ['files']):
        for key in required_keys:
            if cfg.files.get(key) is not None:
                path = os.path.join(root, cfg.files[key])
                if os.path.exists(path):
                    files[key] = path
                else:
                    logging.warning(f'\'{key}\' in \'{path}\' does not exist.')
            else:
                logging.warning(f'Empty \'{key.upper()}\' in config.')

    # Verify required files exist
    check_files(files, required_keys)

    # Process texture with error handling
    try:
        logging.info('Processing texture...')
        texture_path = texture_projection(
            files=files,
            cfg=cfg,
            cfg_resources=cfg_resources,
            uv_sampler=uv_sampler,
            device=device,
            save_root=save_root
        )
        logging.info(f'Texture saved to: {texture_path}')
        return texture_path
    except Exception as e:
        logging.error(f"Texture projection failed: {str(e)}")
        raise

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(format='[%(levelname)s]: %(message)s', level=logging.INFO)
    
    try:
        # Parse arguments
        args = parse()

        # Load configurations
        cfg, cfg_path = load_config(args)
        cfg_resources = get_cfg_defaults(args.resources)

        # Set device with error handling
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{args.gpu}')
            torch.cuda.empty_cache()  # Clear CUDA cache
        else:
            device = torch.device('cpu')
            logging.warning("CUDA not available, using CPU")

        # Process texture
        texture_path = process_texture(args, cfg, cfg_resources, device)

    except Exception as e:
        logging.exception("An error occurred during texture processing:")
        raise
