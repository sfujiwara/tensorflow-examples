from setuptools import find_packages
from setuptools import setup


setup(
    name='trainer',
    packages=find_packages(),
    install_requires=['tensorflow-datasets', 'tensorflow-hub'],
)
