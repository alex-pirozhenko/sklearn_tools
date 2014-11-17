#!/usr/bin/env python

from distutils.core import setup

setup(
      name='sklearn-tools',
      version='0.0.1',
      author='Alex Pirozhenko',
      author_email='alex.pirozhenko@gmail.com',
      # url='http://www.python.org/sigs/distutils-sig/',
      packages=['sklearn_tools', 'sklearn_tools.autoencoder'],
      package_dir={
            'sklearn_tools': 'src/sklearn_tools',
            'sklearn_tools.autoencoder': 'src/sklearn_tools/autoencoder'
      },
)