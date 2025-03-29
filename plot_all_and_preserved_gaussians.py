import torch
import numpy as np
import matplotlib.pyplot as plt



cam = torch.load('../camera_37.pth')['full_proj_transform']
#all_means3D = torch.load('../gparams_fine_frame_000037_stroller_mip_splatting_grad.pth')['means3D']
all_means3D = torch.load('preserved_gaussians.pth')['means3D']
C = torch.cat([all_means3D, torch.ones(len(all_means3D),1)], dim=-1)
pC = (cam.T @ C.unsqueeze(-1)).squeeze()
proj = pC[:,:2]/pC[:,3].unsqueeze(-1)


indices = np.load('indices_to_omit.npy')
all_means3D = torch.load('../gparams_fine_frame_000037_stroller_mip_splatting_grad.pth')['means3D']
omit_means3D = all_means3D[indices]
C_ = torch.cat([omit_means3D, torch.ones(len(omit_means3D),1)], dim=-1)
pC_ = (cam.T @ C_.unsqueeze(-1)).squeeze()
proj_ = pC_[:,:2]/pC_[:,3].unsqueeze(-1)


max_x, max_y = proj_.max(dim=0)[0]
min_x, min_y = proj_.min(dim=0)[0]
nw = torch.cat([min_x[None,None], min_y[None,None]], dim=-1)
sw = torch.cat([min_x[None,None], max_y[None,None]], dim=-1)
ne = torch.cat([max_x[None,None], min_y[None,None]], dim=-1)
se = torch.cat([max_x[None,None], max_y[None,None]], dim=-1)

bbox_all = torch.cat([nw, sw, se, ne, nw], dim=0)

mean = proj_.mean(dim=0)
mean_distance = torch.norm(proj_ - mean.unsqueeze(0), dim=-1).mean().item()

# bounding box from mean
nw_ = torch.cat([mean[0][None,None] - mean_distance, mean[1][None,None] - mean_distance], dim=-1)
sw_ = torch.cat([mean[0][None,None] - mean_distance, mean[1][None,None] + mean_distance], dim=-1)
ne_ = torch.cat([mean[0][None,None] + mean_distance, mean[1][None,None] - mean_distance], dim=-1)
se_ = torch.cat([mean[0][None,None] + mean_distance, mean[1][None,None] + mean_distance], dim=-1)

bbox_mean = torch.cat([nw_, sw_, se_, ne_, nw_], dim=0)

#plt.plot(proj[:,0], proj[:,1], 'r*')
#plt.plot(proj_[:,0], proj_[:,1], 'bo')
#plt.plot(bbox_all[:,0], bbox_all[:,1], 'k')
#plt.plot(bbox_mean[:,0], bbox_mean[:,1], 'm')
#plt.gca().invert_yaxis()

fig, ax = plt.subplots(1,2)
ax[0].plot(proj[:,0], proj[:,1], 'r*')
ax[0].plot(proj_[:,0], proj_[:,1], 'bo')
ax[0].plot(bbox_all[:,0], bbox_all[:,1], 'k')
ax[0].plot(bbox_mean[:,0], bbox_mean[:,1], 'm')
ax[0].invert_yaxis()

ax[1].plot(proj[:,0], proj[:,1], 'r*')
ax[1].plot(bbox_all[:,0], bbox_all[:,1], 'k')
ax[1].plot(bbox_mean[:,0], bbox_mean[:,1], 'm')
ax[1].invert_yaxis()
plt.show()
