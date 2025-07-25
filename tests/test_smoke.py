"""Basic smoke tests for the ARC-TIGERS codebase."""

# Import statements at top level to satisfy linter
import arc_tigers.data.config
import arc_tigers.model.config
import arc_tigers.training.config
from arc_tigers.sampling.utils import get_zero_shot_preds


def test_zero_shot_function_exists():
    """Test that zero-shot function exists and can be imported."""
    assert callable(get_zero_shot_preds)


def test_imports_work():
    """Test that basic imports work."""
    # These are already imported at module level, just verify they exist
    assert arc_tigers.data.config is not None
    assert arc_tigers.model.config is not None
    assert arc_tigers.training.config is not None
