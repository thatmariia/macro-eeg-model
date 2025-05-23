"""
evaluation
-----------
This package contains the (power and coherence) evaluation functions to
compare simulated EEG data against real EEG data.
"""

from .coherence_computer import CoherenceComputer
from .evaluator import Evaluator
from .peak_tester import PeakTester
from .simulation_data_extractor import SimulationDataExtractor
