from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import platform

if platform.system() == 'Linux':
    compilation_args = ['-fopenmp']
    link_args = ['-fopenmp']
elif platform.system() == 'Darwin':
    compilation_args = ['-Xpreprocessor', '-fopenmp', '-lomp']
    link_args = ['-lomp']


ext_modules = [
    Extension(
        "example",
        ["example.pyx"],
        extra_compile_args=compilation_args,
        extra_link_args=link_args,
        include_dirs=[numpy.get_include()],
    )
]




setup(
    ext_modules=cythonize(ext_modules)
)
