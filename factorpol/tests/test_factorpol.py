"""
Unit and regression test for the factorpol package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import factorpol


def test_factorpol_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "factorpol" in sys.modules
