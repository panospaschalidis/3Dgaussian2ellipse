import argparse
import numpy as np
import torch
import math 
import pdb
import json

from conic_color import eval_sh
from render_conic import render


def qvec2rotmat(qvec):
     T1 = torch.cat(
         [(1 - 2 * qvec[:,2]**2 - 2 * qvec[:,3]**2).unsqueeze(-1),
          (2 * qvec[:,1] * qvec[:,2] - 2 * qvec[:,0] * qvec[:,3]).unsqueeze(-1),
          (2 * qvec[:,3] * qvec[:,1] + 2 * qvec[:,0] * qvec[:,2]).unsqueeze(-1)], dim=-1).unsqueeze(1)
     T2 = torch.cat([(2 * qvec[:,1] * qvec[:,2] + 2 * qvec[:,0] * qvec[:,3]).unsqueeze(-1),
               (1 - 2 * qvec[:,1]**2 - 2 * qvec[:,3]**2).unsqueeze(-1),
               (2 * qvec[:,2] * qvec[:,3] - 2 * qvec[:,0] * qvec[:,1]).unsqueeze(-1)], dim=-1).unsqueeze(1)
     T3 = torch.cat([(2 * qvec[:,3] * qvec[:,1] - 2 * qvec[:,0] * qvec[:,2]).unsqueeze(-1),
               (2 * qvec[:,2] * qvec[:,3] + 2 * qvec[:,0] * qvec[:,1]).unsqueeze(-1),
               (1 - 2 * qvec[:,1]**2 - 2 * qvec[:,2]**2).unsqueeze(-1)], dim=-1).unsqueeze(1)
     return torch.cat([T1,T2,T3], dim=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Design 2D covariance matrix')
    parser.add_argument('--params_path', type=str, default='data/gaussians_37.pth')
    parser.add_argument('--camera_path', type=str, default='data/camera_37.json')


    args = parser.parse_args()
    params = torch.load(args.params_path, map_location='cpu')

    means3D = params['means3D']
    scales = params["scales"]
    quaternions = params['rotations']
    opacities = params['opacities']
    shs = params['shs']
   
    with open(args.camera_path, "r") as f:
        cam = json.load(f)
    for key, value in cam.items():
        if isinstance(value, list):
            cam[key] = torch.tensor(value)

    dirs = means3D - cam['camera_center']
    dirs = dirs/torch.norm(dirs, dim=-1).unsqueeze(-1)
    deg = 3
    colors =  eval_sh(deg, shs.permute(0,2,1), dirs)

    tanfovx = math.tan(cam['FoVx'] * 0.5)
    tanfovy = math.tan(cam['FoVy'] * 0.5)
    

    R = qvec2rotmat(quaternions)
    M = R * scales.unsqueeze(1) 
    S = M @ M.permute(0,2,1)
    t = cam['world_view_transform'].T @ torch.cat([
            means3D, 
            torch.ones(len(means3D), 1)
        ], dim=-1).unsqueeze(-1)
    limx = 1.3 * tanfovx

    limy = 1.3 * tanfovy
    
    t = t .squeeze()

    txtz = t[:,0]/t[:,2]

    tytz = t[:,1]/t[:,2]

    max_x = torch.hstack(
            [ 
                txtz[:,None], 
                torch.ones_like(txtz)[:,None]*(-limx) 
            ]
        ).max(dim=-1).values
    min_x = torch.hstack(
            [ 
                max_x[:,None], 
                torch.ones_like(txtz)[:,None]*(limx) 
            ]
        ).min(dim=-1).values
    tx = min_x * t[:,2]
    max_y = torch.hstack(
            [ 
                tytz[:,None], 
                torch.ones_like(tytz)[:,None]*(-limy) 
            ]
        ).max(dim=-1).values
    min_y = torch.hstack(
            [ 
                max_y[:,None], 
                torch.ones_like(tytz)[:,None]*(limy) 
            ]
        ).min(dim=-1).values
    ty = min_y * t[:,2]
    tz = t[:,2]
    focal_y = cam['image_height'] / (2 * tanfovy);
    focal_x = cam['image_width'] / (2 * tanfovx)
    J1 = torch.cat(
            [
                (focal_x/tz)[:,None], 
                torch.zeros_like(t[:,2])[:,None], 
                (-(focal_x*tx)/(tz**2))[:,None]
            ],
            dim=-1).unsqueeze(1)
    J2 = torch.cat(
            [
                torch.zeros_like(t[:,2])[:,None], 
                (focal_y/tz)[:,None], 
                (-(focal_y*ty)/(tz**2))[:,None]
            ],
            dim=-1).unsqueeze(1)
    J3 = torch.zeros_like(J2)
    J = torch.cat([J1,J2,J3], dim=1)
    W = cam['world_view_transform'][:3,:3].T
    T = J @ W
    cov2D = T @ S @ T.permute(0,2,1)
    S2D_ = cov2D[:,:2,:2] + (torch.eye(2)*0.3)
    S2D = S2D_.clone()
    D = torch.det(S2D)
    S2D = S2D.reshape(len(S2D), -1)[:,[0,1,3]]
    conic = (S2D/D[:,None])[:,[2,1,0]]
    conic[:,1] *= -1
    mid = S2D[:,[0,2]].sum(dim=-1)*0.5
    term = torch.hstack(
            [
                (mid**2-D)[:,None], 
                torch.ones_like(mid)[:,None]*(0.01)
            ]
        ).max(dim=-1).values
    l1 = mid + torch.sqrt(term)
    l2 = mid - torch.sqrt(term)
    radius = torch.ceil(
        3*torch.sqrt(torch.hstack([l1[:,None], l2[:,None]]
        ).max(dim=-1).values))
    proj = cam['full_proj_transform'].T @ \
        torch.cat(
            [
                means3D,
                torch.ones(len(means3D),1)
            ], 
            dim=-1
        ).unsqueeze(-1)
    #pdb.set_trace()
    proj = proj[:,[0,1]]/proj[:,3][:,None]
    ndc2pix = lambda v,S : ((v + 1.0) * S - 1.0) * 0.5
    pix2ndc = lambda V,S : (2*V + 1)/S - 1
    
    pix_y = ndc2pix(proj[:,1].clone(), cam['image_height']).squeeze()
    pix_x = ndc2pix(proj[:,0].clone(), cam['image_width']).squeeze()


    compare = lambda grid, a : np.minimum(grid, np.maximum(0,a))
    blk_x = 16
    blk_y = 16
    grid_x = int((cam['image_width'] + blk_x -1)/blk_x)

    grid_y = int((cam['image_height'] + blk_y -1)/blk_y)
    rect_min_x = compare(grid_x, ((pix_x - radius)/blk_x).to(torch.int))
    rect_min_y = compare(grid_y, ((pix_y - radius)/blk_y).to(torch.int))
    rect_max_x = compare(grid_x, ((pix_x + radius + blk_x - 1)/blk_x).to(torch.int))
    rect_max_y = compare(grid_y, ((pix_y + radius +blk_y - 1)/blk_y).to(torch.int))
    flag = (rect_max_x - rect_min_x) * (rect_max_y-rect_min_y) == 0

    eig_dec = torch.linalg.eig(S2D_)
    scales_2D = torch.sqrt(torch.real(eig_dec.eigenvalues))
    rotations_2D = torch.real(eig_dec.eigenvectors)
    scale_coords = scales_2D.unsqueeze(-1) * rotations_2D
    # apart the already normalized rotations bring everythong else to [0,1]
    scales_coords_n = scale_coords[~flag] * torch.tensor([1/(cam['image_width']-1), 1/(cam['image_height']-1)])
    scales_2D_n = np.array(3*torch.norm(scales_coords_n, dim=-1))
    centers_2D_n = np.array(torch.cat([proj[:,0][~flag], proj[:,1][~flag]], dim=-1))
    colors = torch.cat([colors,opacities], dim=-1)[~flag]
    rotations_2D = np.array(rotations_2D[~flag]) 
   
    render(centers_2D_n,colors, scales_2D_n, rotations_2D, cam) 
