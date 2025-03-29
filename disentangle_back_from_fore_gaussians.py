import torch

import numpy as np

from PIL import Image

from torchvision.transforms import ToTensor

transform = ToTensor()
flag_37 = transform(Image.open('/media/atlas/datasets/DAVIS-2017-trainval-testdev-Full/DAVIS/Annotations/Full-Resolution/stroller/00037.png'))

# background gaussians camera 37
Y, X = torch.where(flag_37.squeeze()==0)

indices = Y*1920 + X

gindices_ = []

contrib_37 = torch.load('../save_for_later_fine_frame_000037.pth')['contrib']
from tqdm import tqdm
tbar = tqdm(range(len(indices.tolist())))
for index in indices.tolist():
    gindices_.extend(contrib_37[index][1:])
    tbar.update(1)

# foreground gaussians camera 37
Y, X = torch.where(flag_37.squeeze()!=0)

indices = Y*1920 + X
tbar = tqdm(range(len(indices.tolist())))
gindices = []
for index in indices.tolist():
    gindices.extend(contrib_37[index][1:])
    tbar.update(1)

fore_G = np.unique(gindices)

back_G = np.unique(gindices_)

fore_G = list(fore_G)

back_G = list(back_G)
backup_back_G = back_G.copy()
backup_fore_G = fore_G.copy()

print(len(fore_G))
print(len(back_G))
for el in fore_G:
    #print(f'element is {el}')
    try:
        index = back_G.index(el)
    #print(f'index is {index}')
        back_G.pop(index)
    except ValueError as e:
        continue

for el in backup_back_G:
    #print(f'element is {el}')
    try:
        index = fore_G.index(el)
    #print(f'index is {index}')
        fore_G.pop(index)
    except ValueError as e:
        continue
print(len(fore_G))
print(len(back_G))

params = torch.load('../gparams_fine_frame_000037_.pth')

A = back_G
torch.save(back_G, 'background_37_g_indices.pth')
torch.save({'means3D': params['means3D'][A],
'scales': params['scales'][A],
'rotations': params['rotations'][A],
'shs': params['shs'][A],
'opacities': params['opacities'][A]}, 'gaussians_37_back.pth')

A = fore_G

torch.save(fore_G, 'foreground_37_no_back_overlap_g_indices.pth')
torch.save({'means3D': params['means3D'][A],
'scales': params['scales'][A],
'rotations': params['rotations'][A],
'shs': params['shs'][A],
'opacities': params['opacities'][A]}, 'gaussians_37_fore_no_back_overlap.pth')


A = backup_fore_G

torch.save(backup_fore_G, 'foreground_37_back_overlap_g_indices.pth')
torch.save({'means3D': params['means3D'][A],
'scales': params['scales'][A],
'rotations': params['rotations'][A],
'shs': params['shs'][A],
'opacities': params['opacities'][A]}, 'gaussians_37_fore_back_overlap.pth')





#flag_67 = transform(Image.open('/media/atlas/datasets/DAVIS-2017-trainval-testdev-Full/DAVIS/Annotations/Full-Resolution/stroller/00067.png'))
#
#contrib_67 = torch.load('../save_for_later_fine_frame_000067.pth')['contrib']
#
#
#
#
## foreground gaussians camera 37
#Y, X = torch.where(flag_67.squeeze()!=0)
#
#indices = Y*1920 + X
#
#gindices = []
#
#tbar = tqdm(range(len(indices.tolist())))
#for index in indices.tolist():
#    gindices.extend(contrib_67[index][1:])
#    tbar.update(1)
#
## background gaussians camera 67
#Y, X = torch.where(flag_67.squeeze()==0)
#
#indices = Y*1920 + X
#
#gindices_ = []
#
#tbar = tqdm(range(len(indices.tolist())))
#for index in indices.tolist():
#    gindices_.extend(contrib_67[index][1:])
#    tbar.update(1)
#
#
#fore_G = np.unique(gindices)
#back_G = np.unique(gindices)
#
#fore_G = list(fore_G)
#
#back_G = list(back_G)
#
#for el in fore_G:
#    #print(f'element is {el}')
#    try:
#        index = back_G.index(el)
#    #print(f'index is {index}')
#        back_G.pop(index)
#    except ValueError as e:
#        continue
#
#
#
#A = back_G
#
#torch.save({'means3D': params['means3D'][A],
#'scales': params['scales'][A],
#'rotations': params['rotations'][A],
#'shs': params['shs'][A],
#'opacities': params['opacities'][A]}, 'gaussians_67_back.pth')
#
#
#
#A = fore_G
#
#torch.save({'means3D': params['means3D'][A],
#'scales': params['scales'][A],
#'rotations': params['rotations'][A],
#'shs': params['shs'][A],
#'opacities': params['opacities'][A]}, 'gaussians_67_fore.pth')






