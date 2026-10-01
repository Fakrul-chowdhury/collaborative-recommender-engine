"""Implicit-feedback recommender for retail transactions (ALS + content-based cold start)."""

import os

# implicit's ALS parallelises across users itself; a multi-threaded BLAS underneath
# oversubscribes the CPU and slows training down. Must be set before numpy is imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
