from setuptools import setup

__version__ = 'unknown'
for line in open('acoustics_hardware/__init__.py', mode='r', encoding='utf-8'):
    if line.startswith('__version__'):
        exec(line)
        break


setup(name='acoustics-hardware',
      version=__version__,
      description='Controlling hardware used in acoustic measurement systems',
      long_description=open('README.rst', mode='r', encoding='utf-8').read(),
      long_description_content_type='text/x-rst',
      url='https://github.com/AppliedAcousticsChalmers/acoustics-hardware',
      author='Carl Andersson',
      author_email='carl.andersson@chalmers.se',
      license='MIT',
      packages=['acoustics_hardware'],
      install_requires=[
          'h5py',
          'matplotlib',
          'nidaqmx;platform_system=="Windows"',
          'numpy',
          'pyserial',
          'SchunkMotionProtocol',
          'scipy',
          'sounddevice',
      ],
      )
