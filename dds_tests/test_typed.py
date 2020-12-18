import dds


@dds.dds_function("/test")
def f() -> str:
    return ""


def test_typed_annotation():
    x = f()
    z: int = x
