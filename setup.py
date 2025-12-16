#!/usr/bin/env python

from setuptools import setup

setup(name='ycbev_toolkit',
      version='1.1.0',
      description='A Python toolkit for loading and processing event data from the YCB-Ev dataset.',
      author='Pavel Rojtberg',
      url='https://github.com/paroj/ycbev',
      py_modules=['ycbev'],
      license='MIT',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      install_requires=['zstandard', 'numpy'],
)