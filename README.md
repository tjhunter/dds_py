# dds_py - Data driven software

Data-driven software (python implementation)

## Introduction

The DDS package solves the synchronization problem between code and data. It allows programmers,
 scientists and data scientists to integrate code with data and data with code without fear of
 stale data, disparate storage frameworks or concurrency issues. DDS allows quick collaboration and 
 data software reuse without the complexity. In short, you do not have to think about changes in your data pipelines.


## How to use

This package is published on PyPI:

```
pip install dds_py
```

This package is known to work on python 3.6, 3.7, 3.8. No other versions are officially supported. Python 3.4 and 3.5 might work but they are not supported.

__Plotting dependencies__ If you want to plot the graph of data dependencies, you must install separately the `pydotplus` package, which requires `graphviz` on your system to work properly. Consult the documentation of the `pydotplus` package for more details. The `pydotplus` package is only required with the `dds_export_graph` option.

__Databricks users:__ If you want to use this package with Databricks, some specific hooks for Spark are available. See this notebook for a complete example:

https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/7816249071411394/4492656151009213/5450662050199348/latest.html

## Documentation

API reference, tutorials and FAQs are located here: https://tjhunter.github.io/dds_py/

## Example
 
Here is the Hello world example (using type annotations for clarity)

```python
import dds
import requests 

@dds.data_function("/hello_data")
def data() -> str:
  url = "https://gist.githubusercontent.com/bigsnarfdude/515849391ad37fe593997fe0db98afaa/raw/f663366d17b7d05de61a145bbce7b2b961b3b07f/weather.csv"
  return requests.get(url=url, verify=False).content.decode("utf-8")

data()
```
This example does the following:
- it defines a source of data, here a piece of weather data from the internet. This source is defined as the function `data_creator`
- it assigns the data produced by this source into a variable (`data`) and also to a path in a storage system (`/hello_data`) 

The DDS library guarantees the following after evaluation of the code:
1. the path `/hello_data` contains a copy of the data returned by `data_creator`, as if the function `data_creator` had been called at this moment
2. the function `data_creator` is only evaluated when its inputs, or its code, are modified (referential transparency)

## License

The `dds` package is published under the Affero General Public License.
