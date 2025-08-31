"""
Basic test to verify test setup is working.
"""
import pytest


def test_basic_math():
    """Test that basic math works - sanity check."""
    assert 1 + 1 == 2


def test_pytest_markers():
    """Test that pytest markers are working."""
    assert True


@pytest.mark.unit
def test_unit_marker():
    """Test with unit marker."""
    assert True


def test_imports():
    """Test that we can import required modules."""
    import asyncio
    import json
    import tempfile
    
    assert asyncio is not None
    assert json is not None
    assert tempfile is not None


def test_test_data_generator():
    """Test that our test data generator works."""
    from tests.utils.test_data_generator import TestDataGenerator
    
    generator = TestDataGenerator()
    users = generator.generate_user_data(2)
    
    assert len(users) == 2
    assert all("email" in user for user in users)
    assert all("password" in user for user in users)