

import numpy as np
import torch
import cv2
import os
import open3d as o3d
import trimesh
from tqdm import tqdm
from lib.utils.uv_sample.divided_uv_generator import Index_UV_Generator
from lib.utils.mesh_util import load_obj, save_mtl
from lib.utils.image_util import write_pic, inverse_mask
from lib.tools.feature_projection import orthogonal, index
from lib.utils.smplx_util import back_to_econ_axis
from lib.utils.config import  get_cfg_defaults
from lib.utils.common_util import check_key



def ray_cast(mesh, rays):
    mesh_scene = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_scene)
    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    ans = scene.count_intersections(rays)
    front_orient = ans.numpy() > 1
    return front_orient

def ray_cast_trimesh(mesh):
    vertices = np.asarray(mesh.vertices)
    front_orient = np.zeros(len(vertices))
    
    pbar = tqdm(range(len(vertices)))
    pbar.set_description('Visible')
    for i in pbar:
        intersection = mesh.ray.intersects_id([vertices[i] + np.asarray([0., 0., 1e-7])], [[0., 0., 1.]],
                                              multiple_hits=True,
                                              return_locations=False)
        front_orient[i] = len(intersection[0])
    return front_orient != 0

def get_calib(param, img_size):
    ortho_ratio = param.get('ortho_ratio')
    scale = param.get('scale')
    center = param.get('center')
    R = param.get('R')
    translate = -np.matmul(R, center).reshape(3, 1)
    extrinsic = np.concatenate([R, translate], axis=1)
    extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
    scale_intrinsic = np.identity(4)
    scale_intrinsic[0, 0] = -scale / ortho_ratio
    scale_intrinsic[1, 1] = -scale / ortho_ratio
    scale_intrinsic[2, 2] = scale / ortho_ratio
    uv_intrinsic = np.identity(4)
    uv_intrinsic[0, 0] = 1.0 / float(img_size // 2)
    uv_intrinsic[1, 1] = 1.0 / float(img_size // 2)
    uv_intrinsic[2, 2] = 1.0 / float(img_size // 2)
    # Transform under image pixel space
    trans_intrinsic = np.identity(4)
    intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
    calib = np.matmul(intrinsic, extrinsic).astype(float)
    return calib, R, translate

def texture_projection(files, cfg, cfg_resources, uv_sampler, device, save_root=None, camera_param_path=None):
    try:
        # Load mesh
        vertices, _, faces = load_obj(files['completed_mesh'])
        if vertices.size == 0 or faces.size == 0:
            raise ValueError("Invalid mesh data")
        vertices = vertices.astype(np.float32)
        vertices_tensor = torch.from_numpy(vertices).to(device)
        
        # Load and verify image
        img = cv2.imread(files['image'])
        if img is None:
            raise ValueError(f"Could not load image from {files['image']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(dim=0).to(device)
        
        # Load and verify mask
        mask = cv2.imread(files['mask'])
        if mask is None:
            raise ValueError(f"Could not load mask from {files['mask']}")
        mask = mask.astype(np.float32) / 255.
        mask = torch.from_numpy(mask).float().permute(2, 0, 1).unsqueeze(dim=0).to(device)
        
        # Load UV mask
        tex_mask = cv2.imread(cfg_resources.masks.mask_wo_eyes)
        if tex_mask is None:
            raise ValueError("Could not load UV mask")
        tex_mask = tex_mask.astype(np.float32) / 255.
        tex_mask = cv2.flip(tex_mask, 0)
        tex_mask_tensor = torch.from_numpy(tex_mask).to(device)
        
        # Load calibration
        if camera_param_path is not None:
            param = np.load(camera_param_path, allow_pickle=True).item()
            vertices_pro = (vertices - param['center']) * param['scale'] / 100
            vertices_pro = np.matmul(param['R'], vertices_pro.transpose(1, 0)).transpose(1, 0)
        else:
            vertices_pro = vertices.copy()
            
        calib = np.load(cfg_resources.econ_calib)
        calib = torch.from_numpy(calib).float().unsqueeze(dim=0).to(device)
        
        # Perform projection
        tex, mask, colors = projection(
            vertices=vertices_tensor,
            faces=faces,
            img=img,
            img_mask=mask,
            tex_mask=tex_mask_tensor,
            calib=calib,
            sampler=uv_sampler,
            device=device,
            dilate_iter=3,
            front_only=False
        )
        
        if tex is None or mask is None or colors is None:
            raise ValueError("Projection failed")

        if save_root is not None:
            os.makedirs(save_root, exist_ok=True)
            tex_path = os.path.join(save_root, 'partial_tex.png')
            mask_path = os.path.join(save_root, 'partial_mask.png')
            mesh_path = os.path.join(save_root, 'partial_colored.obj')
            mtl_path = os.path.join(save_root, 'texture.mtl')
            
            cv2.imwrite(tex_path, tex * 255)
            cv2.imwrite(mask_path, mask * 255)
            
            _, vts, _ = load_obj(cfg_resources.models.smplx_vts_template)
            save_mtl(vertices, faces, vts, mesh_path, mtl_path, 'partial_tex.png', colors=colors)
            
            return tex_path, mask_path, mesh_path
            
        return tex, mask, colors
        
    except Exception as e:
        print(f"Error in texture_projection: {str(e)}")
        raise

def projection(vertices, faces, img, img_mask, tex_mask, calib, sampler, device, dilate_iter=3, front_only=False):
    try:
        # Create trimesh object
        mesh = trimesh.Trimesh()
        mesh.faces = faces[..., 0] - 1
        vertices_econ = vertices.cpu().numpy() if torch.is_tensor(vertices) else vertices
        mesh.vertices = vertices_econ

        # Project vertices
        vertices_econ_tensor = torch.from_numpy(vertices_econ).float().unsqueeze(dim=0).permute(0, 2, 1).to(device)

        # Get projected coordinates
        crop_query_points = orthogonal(vertices_econ_tensor, calib, None)
        if crop_query_points is None:
            raise ValueError("Failed to compute orthogonal projection")
            
        xy = crop_query_points[:, :2, :]
        z = crop_query_points[:, 2:3, :]

        # Normalize coordinates to [-1, 1] range
        xy = torch.clamp(xy, min=-1.0, max=1.0)

        # Sample features with proper error handling
        mask_fea = index(img_mask, xy)
        if mask_fea is None:
            raise ValueError("Failed to sample mask features")
        mask_fea = mask_fea.permute(0, 2, 1)[0].cpu().numpy().sum(axis=-1)
        mask_fea[mask_fea!=0] = 1

        sample_fea = index(img, xy)
        if sample_fea is None:
            raise ValueError("Failed to sample image features")
        sample_fea = sample_fea.permute(0, 2, 1)[0].cpu().numpy()

        # Handle occlusions
        if front_only:
            front_orient = ray_cast_trimesh(mesh)
            front_orient[mask_fea==0] = True
            sample_fea[front_orient] = 1.
        else:
            sample_fea[mask_fea==0] = 1.

        # Generate UV map
        sample_fea_tensor = torch.from_numpy(sample_fea).unsqueeze(dim=0).to(device)
        partial_tex = sampler.get_UV_map(sample_fea_tensor)
        if partial_tex is None:
            raise ValueError("Failed to generate UV map")
        partial_tex = partial_tex.cpu().numpy()[0].astype(np.float32)

        # Process texture maps
        if front_only:
            sample_fea[front_orient] = 2.
        else:
            sample_fea[mask_fea==0] = 2.

        partial_tex_diff = sampler.get_UV_map(torch.from_numpy(sample_fea).unsqueeze(dim=0).to(device))
        partial_tex_diff = partial_tex_diff.cpu().numpy()[0].astype(np.float32)

        diff = partial_tex - partial_tex_diff
        diff[diff != 0] = 1
        partial_tex[diff==1] = 0

        # Flip and process masks
        partial_tex = cv2.flip(partial_tex, 0)
        partial_tex_mask = inverse_mask(partial_tex)

        # Ensure tex_mask is on correct device and format
        if torch.is_tensor(tex_mask):
            tex_mask = tex_mask.cpu().numpy()

        # Apply masks with proper bounds checking
        partial_tex[partial_tex_mask==1] = 1
        partial_tex[tex_mask == 0.] = 0.
        partial_tex_mask[tex_mask == 0.] = 0.

        # Dilate mask
        kernel = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]]).astype(np.uint8)
        partial_tex_mask = cv2.dilate(partial_tex_mask.astype(np.uint8), kernel, iterations=dilate_iter)
        partial_tex_mask = cv2.cvtColor(partial_tex_mask, cv2.COLOR_RGB2GRAY)
        partial_tex_mask = cv2.cvtColor(partial_tex_mask, cv2.COLOR_GRAY2RGB)

        # Final processing
        partial_tex[partial_tex_mask==1] = 0
        partial_tex = cv2.cvtColor(partial_tex, cv2.COLOR_RGB2BGR)

        # Add debug prints
        print(f"Partial tex range: [{partial_tex.min()}, {partial_tex.max()}]")
        print(f"Mask range: [{partial_tex_mask.min()}, {partial_tex_mask.max()}]")
        print(f"Sample features range: [{sample_fea.min()}, {sample_fea.max()}]")

        return partial_tex, partial_tex_mask, sample_fea

    except Exception as e:
        print(f"Error in projection function: {str(e)}")
        return None, None, None