# -*- coding: utf-8 -*-
"""Test data used in planning pre-tests and helper function validation"""

import numpy as np
from .pathplanning_utils import (Vertex, get_neighborhood)


NAV_MAP = np.rot90(np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                             [1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                             [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                             [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                             [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                             [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]), 3)
START = Vertex(4, 4)
END = Vertex(4, 7)
NEIGHBORS = get_neighborhood(8)
