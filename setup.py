from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

required = filter(str.strip, required)
required = filter(lambda x: not x.startswith('-e .'), required)

setup(
    name="src",
    version="0.1.0",
    packages=find_packages(),
    install_requires=list(required),
)