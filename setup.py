#!/usr/bin/env python

from distutils.core import setup

setup(name='Bangla POS Tagger',
      version='1.0',
      description='Bangla Based POS Tagger',
      author='Abhishek Gupta',
      author_email='abhishekgupta.iitd@gmail.com',
      url='http://github.com/abhishekgupta92/bangla_pos_tagger',
      py_modules=['bangla_pos_tagger'],
      package_dir={'data':'data'},
      package_data={'data': ['*']},
      data_files=[('config',['params.py'])],
     )
