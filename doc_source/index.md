# DDS - Data Driven Software

This is the main documentation page for the `dds` package.

__Why DDS?__
The `dds` package solves the synchronization problem between code and data. It allows programmers,
 scientists and data scientists to integrate code with data and data with code without fear of
 stale data, disparate storage frameworks or concurrency issues. DDS allows quick collaboration and 
 data software reuse without the complexity. In short, you do not have to think about changes in your 
data pipelines.


## Requirements

`dds` is officially supported for Python versions 3.6 to 3.9. 

`dds` has one requirement:

 - asttokens
 
In addition, the following dependency provides support for plotting the 
graph of calculations:

 - pydotplus
 
The following Python frameworks are provided out of the box:

 - Apache Spark
 - pandas
 
The following notebook systems have integration points:

 - Jupyter
 - Databricks notebooks

## Installation

When using PyPI:

```sh
pip install dds_py
```

## Python support

Python is a very rich language, and not all features of Python
are useful for data scientists. 
In general, `dds` takes an opinionated approach about the usefulness of specific 
Python features. All the essential Python code constructs are supported by `dds`:

 - built-in types (primitives, dictionaries, lists, ...)
 - functions, modules, notebook-defined functions, lambda functions, ...
 - classes (without subclasses)

The following features are not expected to be supported:

 - async functions
 - method reassignment
 - advanced machinery to generate code at runtime

The following features might be added to a feature release:

 - decorators
 - static and class functions
 - futures, concurrency, threading, manual pickling


## License

`dds` is released under the Affero General Public License (AGPL).




