from setuptools import setup
import sys
import os.path


sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'acoustics_hardware'))
from _version import version_manager  # We cannot import the _version module, but we can import from it.

with version_manager() as version:
    setup(
        name='acoustics-hardware',
        version=version,
        description='Controlling hardware used in acoustic measurement systems',
        long_description=open('README.rst').read(),
        long_description_content_type='text/x-rst',
        url='https://github.com/AppliedAcousticsChalmers/acoustics-hardware',
        author='Carl Andersson',
        author_email='carl.andersson@chalmers.se',
        license='MIT',
        packages=['acoustics_hardware'],
        install_requires=[
            'numpy',
            'scipy',
            'sounddevice',
            'nidaqmx;platform_system=="Windows"',
            'pyserial',
            'SchunkMotionProtocol',
            'zarr',
        ],
    )
