from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='conv4d_cuda',
    ext_modules=[
        CUDAExtension('conv4d_cuda', [
            'conv4d.cpp',
            'conv4d_kernel.cu',
            'cost_volume.cu',
            ], extra_compile_args={'cxx': ['-g'], 'nvcc': ['-gencode=arch=compute_60,code=compute_60','-O2']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
