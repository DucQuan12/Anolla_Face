from Cython.Build import cythonize
from distutils.core import Extension, setup

ext = Extension(name="FaceID", sources=["App.py"])

setup(ext_modules=cythonize(ext))