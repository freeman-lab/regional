#!/usr/bin/env python

from setuptools import setup

version = '1.1.0'

setup(
    name='regional',
    version=version,
    description='simple manipualtion and display of spatial regions',
    author='freeman-lab',
    author_email='the.freeman.lab@gmail.com',
    packages=['regional'],
    url='https://github.com/freeman-lab/regional',
    install_requires=open('requirements.txt').read().split(),
    long_description='See https://github.com/freeman-lab/regional'
)