from setuptools import setup, find_packages

setup(
    name='src',
    version='0.1',
    description='acouslic_src',
    packages=find_packages(),
    install_requires=[
        'lightning',
        'hydra-core'],
)