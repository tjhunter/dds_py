from dds.structures import KSException
import pytest
from .utils import cleandir, unreachable
from dds_tests.unauthorized_mod.unauthorized import my_fun

_ = cleandir


@pytest.mark.usefixtures("cleandir")
def test_1():
    with pytest.raises(KSException) as e:
        my_fun()
    assert "trieved, howe" in str(e.value), e
