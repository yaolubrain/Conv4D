from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='conv4d.cpp',
    ext_modules=[
        CppExtension('conv4d_cpp', ['conv4d.cpp', 'cost_volume.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
