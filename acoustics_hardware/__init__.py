import logging

__version__ = '0.1.0'

logging.VERBOSE = 15
logging.FRAMES = 5
logging.addLevelName(logging.FRAMES, 'FRAMES')
logging.addLevelName(logging.VERBOSE, 'VERBOSE')

class AcousticsHardwareLogger(logging.Logger):
	def verbose(self, *args, **kwargs):
		self.log(logging.VERBOSE, *args, **kwargs)

	def frames(self, *args, **kwargs):
		self.log(logging.FRAMES, *args, **kwargs)

logging.setLoggerClass(AcousticsHardwareLogger)
logger = logging.getLogger(__name__)

__all__ = ['core',
           'devices',
           'triggers',
           'generators',
           'processors',
           'distributors',
           'serial',
           ]

from . import *

logging.setLoggerClass(logging.Logger)
log_fmt_spec = '%(asctime)s - %(name)s - %(levelname)s - %(threadName)s: %(message)s'
