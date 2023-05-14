Acoustics Hardware
==================

This Python package provides a unified interface to several hardware platforms used in acoustic measurements, e.g. audio interfaces.
The purpose of this package is twofold, both to enable cross-hardware interactions with a consistent interface, and to simplify scripting of advanced measurement setups.

Documentation:
    http://acoustics-hardware.readthedocs.io/
Source code and issue tracker:
    https://github.com/AppliedAcousticsChalmers/acoustics-hardware


Installation
------------

For convenience reasons we highly recommend to use `Conda`_ (miniconda for simplicity) to manage your Python installation.
Once installed, you can use the following steps to receive and use this branch of the toolbox.

| Clone (or download) this branch of the repository:
| ``git clone -b add-notebook-SMA-example https://github.com/AppliedAcousticsChalmers/acoustics-hardware.git``
| ``cd acoustics-hardware/``

| Create a `Conda`_ environment with the toolbox and the required dependencies:
| ``conda env create --file environment.yml --force``

| Activate the environment:
| ``conda activate acoustics-hardware-dev``


Examples
--------

Use the following steps to run the provided `Jupyter`_ notebook examples to perform acoustic measurements.

| Start the `Jupyter`_ server and open the desired example, e.g.:
| ``jupyter notebook examples/measurement_single_microphone.ipynb``
| ``jupyter notebook examples/measurement_DH_on_turntable.ipynb.ipynb``
| ``jupyter notebook examples/measurement_sequential_SMA.ipynb.ipynb``


.. _Conda: https://conda.io/en/master/miniconda.html
.. _Jupyter: https://jupyter.org/
