from .state import QuantumState
from .circuit import QuantumCircuit
from . import gates
from . import algorithms

# matplotlib is optional -- core sim works without it
try:
    from . import visualization  # noqa: F401
    _HAS_VIS = True
except ImportError:
    _HAS_VIS = False

__all__ = ["QuantumState", "QuantumCircuit", "gates", "algorithms"]
if _HAS_VIS:
    __all__.append("visualization")

__version__ = "0.1.0"
