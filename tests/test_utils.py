import pytest
from llmeter.utils import DeferredError


def test_deferred_error_on_import():
    # Simulate the import attempt
    try:
        import obscurelib  # type: ignore
    except ImportError as e:
        obscurelib = DeferredError(e)

    # Verify that obscurelib is now a DeferredError instance
    assert isinstance(obscurelib, DeferredError)

    # Attempt to use the 'obscurelib'
    with pytest.raises(ImportError) as exc_info:
        obscurelib.some_function()

    # Verify that the original ImportError is raised
    assert "No module named 'obscurelib'" in str(exc_info.value)

    # Try to access an attribute
    with pytest.raises(ImportError) as exc_info:
        _ = obscurelib.some_attribute

    # Verify that the original ImportError is raised
    assert "No module named 'obscurelib'" in str(exc_info.value)
