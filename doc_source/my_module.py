import dds


@dds.data_function("/my_function")
def my_function():
    return "my_function"