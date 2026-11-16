"""Make symbolic planning reproducible across processes.

The HW3 A* planner (unchanged) iterates over Python sets of string tuples, and
Python randomises string hashing per process, so ties between equal-cost plans
broke differently on every run. The simulator and the hardware-plan capture
must produce the SAME plan, so both re-launch themselves with a fixed hash seed.
"""
import os
import sys

HASH_SEED = '0'


def ensure_deterministic_hashing():
    if os.environ.get('PYTHONHASHSEED') != HASH_SEED:
        os.environ['PYTHONHASHSEED'] = HASH_SEED
        os.execv(sys.executable, [sys.executable] + sys.argv)
