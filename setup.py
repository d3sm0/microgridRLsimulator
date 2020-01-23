from setuptools import setup, find_packages

packages_ = find_packages()
packages = [p for p in packages_ if not (p == 'tests')]

with open("README.md", 'r') as f:
    long_description = f.read()

setup(name='microgridRLsimulator',
      version='0.11dev',
      description='Simulator for MicroGrid Control via Reinforcement Learning',
      long_description=long_description,
      keywords='environment, agents, rl, openaigym, gym, energy',
      url='https://github.com/bcornelusse/microgridRLsimulator',
      author='',
      author_email='',
      license='BSD 2-Clause',
      packages=packages,
      install_requires=[
          'matplotlib',
          'numpy',
          'pandas',
          'sphinx',
          'gym'
      ],
      zip_safe=False)
