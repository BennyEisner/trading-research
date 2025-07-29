#!/usr/bin/env python3

"""
LSTM Performance BAseline Framework
Standalone LSTM performance before ensemble integration
"""


import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precisio_scre, recall_score
from sklearn.model_selection import TimeSeriesSplit
