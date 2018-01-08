from setuptools import setup

# TODO: Add github url
# TODO: Add author
# TODO; Add author email
# TODO: Add licence


setup(
    name='acoustics-hardware',
    version='0.0.1',
    description='Controlling hardware used in acoustic measurement systems',
    packages=['acoustics_hardware'],
    install_requires=[
        'numpy',
        'scipy'],
    extras_require={
        'hdf': ['h5py'],
        'ni': ['nidaqmx;platform_system=="Windows"'],
        'serial': ['pyserial'],
        'motors': ['SchunkMotionProtocol']
    })
