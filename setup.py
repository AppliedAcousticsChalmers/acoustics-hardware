from setuptools import setup

__version__ = 'unknown'
for line in open('acoustics_hardware/__init__.py'):
    if line.startswith('__version__'):
        exec(line)
        break


setup(name='acoustics-hardware',
      version=__version__,
      description='Controlling hardware used in acoustic measurement systems',
      long_description=open('README.RST').read(),
      url='https://github.com/CarlAndersson/acoustics-hardware',
      author='Carl Andersson',
      author_email='carl.andersson@chalmers.se',
      license='MIT',
      packages=['acoustics_hardware'],
      install_requires=[
          'numpy',
          'scipy',
          'h5py',
          'sounddevice',
          'nidaqmx;platform_system=="Windows"',
          'pyserial',
          'SchunkMotionProtocol'],
      )
