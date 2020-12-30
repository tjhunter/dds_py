import dds

# Function that is still labeled a data function but that is not
# accessible in the packages.
@dds.data_function("/my_fun")
def my_fun():
    return 1
