## 3Dgaussian2ellipsis

`3D Gaussian Splatting` has emerged as a powerful framework for novel 
view synthesis. 3D Gaussians are splatted to the defined by camera parameters image
plane and the gaussian primitives are optimized as described in [GS](https://github.com/graphdeco-inria/gaussian-splatting). Concerning
the spaltting procedure and the alpha decomposition of the employed Gaussians we are able 
to visualize the outcome but not the topology of the splatted Gaussians. That is why we created 
this useful tool in `moderngl` to facilitate `2D splatted Gaussians/ellipses` visualization on the image plane.


## Installation
```
conda create --name ellipsis python=3.10
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install moderngl==5.12.0
pip install matplotlib
```
In case you encounter the followuing error
```libGL error: MESA-LOADER: failed to open swrast: /home/your_username/anaconda3/envs/ellipsis/lib/python3.10/site-packages/numpy/_core/../../../../libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libLLVM-15.so.1) (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: swrast
```
navigate to environment directory
```
cd anaconda3/envs/ellipis
```
and do the following 
```
rm -rf libstd.6.0.29
```
error occurs due to inconsistency between libstd.6.0.29 library and GLIBCXX_3.4.30

## PreProcessing

## Run

## Examples

