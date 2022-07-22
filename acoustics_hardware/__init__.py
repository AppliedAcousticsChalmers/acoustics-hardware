import logging
logger = logging.getLogger(__name__)

from . import _version  # noqa: E402
__version_info__ = _version.version_info
__version__ = _version.version
del _version

from . import interfaces, generators, recorders, analysis, signal_tools, timestamps, triggers  # noqa: E402, F401
