import sys
from pyfiglet import Figlet
import logging
from pkg_resources import get_distribution

__version__ = get_distribution('robopilot').version

logging.basicConfig(level=logging.INFO)
f = Figlet(font='speed')


print(f.renderText('Robopilot'))
print(f'using robopilot v{__version__} ...')

if sys.version_info.major < 3 or sys.version_info.minor < 6:
    msg = f'Robopilot Requires Python 3.6 or greater. You are using {sys.version}'
    raise ValueError(msg)

# The default recursion limits in CPython are too small.
sys.setrecursionlimit(10**5)

from .vehicle import Vehicle
from .memory import Memory
from . import utils
from . import config
from . import contrib
from .config import load_config
