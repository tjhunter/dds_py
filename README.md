# dds_py - Data driven software

Data-driven software (python implementation)

## Introduction

The DDS package solves the synchronization problem between code and data. It allows programmers,
 scientists and data scientists to integrate code with data and data with code without fear of
 stale data, disparate storage frameworks or concurrency issues. DDS allows quick collaboration and 
 data software reuse without the complexity.
 
In the world of data-driven software, executing code leads to the creation of _data artifacts_, which can be 
of any sort and shape that the work requires:
- _datasets_ : collections of data items
- _models_ : compact representations of datasets for specific tasks (classification, ...)
- _insights_: information about datasets and models that provide human-relatable cues about other artifacts

Combining software with data is currently a hard challenge, because existing programming paradigms
aim at being universal and are not tuned to the specific challenges of combining data and code 
within a single product. DDS provides the low-level foundations to do that, in the spirit
of Karparthy's Software 2.0 directions (TODO: cite). `dds_py` is a software implementation of these ideas

Here is the Hello world example (using type annotations for clarity)

```python
import dds
import requests 

def data_creator() -> str:
  return requests.get("hello_world")

data: str = dds.keep("/hello_data", data_creator)
```
This example does the following:
- it defines a source of data, from the internet. This source is defined as the function `data_creator`
- it assigns the data produced by this source into a variable (`data`) and also to a path in a storage system (`/hello_data`) 

The DDS library guarantees the following after evaluation of the code:
1. the path `/hello_data` contains a copy of the data returned by `data_creator`, as if the function `data_creator` had been called at this moment
2. the function `data_creator` is only evaluated when its inputs, or its code, are modified (referential transparency)

This model has profound consequences for the programmers:
- computationally expensive data functions (such as building models) can be composed and built upon _very cheaply_, as if they were
variables. DDS alleviates the need to decompose data pipelines into multiple stages because of technological requirments. 

At its core, the programming model of DDS is very simple:
- functions are assumed to be idempotent, if not pure
- functions are referentially transparent (they can be replaced with their output)
- artifacts of any sort (models, data, statistics) are stored in a central repository
- the programming model is assumed to be hermetic (only the I/O tracked within the framework is expected to happen)
