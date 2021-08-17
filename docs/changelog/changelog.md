# Changelog

## v0.11.0

This release adds configuration options and addresses performance issues 
introduced by v0.10. In addition, some groundwork started to introduce more limits 
around the hashing of objects, such as big lists, into the signature. This part is 
not enabled yet.

* [GH-136](https://github.com/tjhunter/dds_py/issues/136) Performance issues on deeply nested code.
This was a regression from v0.10 caused by a sub-optimal algorithm.

* [GH-137](https://github.com/tjhunter/dds_py/issues/137) Introduction of the configuration framework.
This allows the behaviour of some specific aspects to be finetuned. The feature is currently left undocumented
until more options are added.

## v0.10.0

This release adds two bugfixes. 

* [GH-130](https://github.com/tjhunter/dds_py/issues/130) Failing for overlapping paths. For example, `dds` used to accept an evaluation with both `/f` and `/f/g` paths defined.
Such a structure is ill-defined for most filesystems. It is now an error.

* [GH-133](https://github.com/tjhunter/dds_py/issues/133) Higher-order function calls are not properly captured. For example, the python code `map(some_function, range(2))`
used to miss `some_function` as a dependency. Such a function is now accounted for. This will retrigger signature 
calculations in this corner case.

## v0.9.0

Adding two useful stores for checking the correctness of the code without
relying on DDS. See the documentation of `dds.set_store`.

## v0.8.0

A number of small improvements in ergonomics to this release:

* the error messages are more complete and include more contextual information
* more types are supported by default during the analysis phase: lists, dictionaries,
dates (`datetime` objects), arbitrary named tuples and arbitrary data classes.
* the input for `@data_function` has been tightened to reflect the fact that data functions
should not take arguments (`dds.keep` should be used instead). Passing arguments
now triggers an error.

## v0.7.3

Fixes the usage of positional and keyworded arguments when used in conjunction
with `dds.keep`.

## v0.7.2

Small usability fixes in this release:

* delaying the creation of a default store (and all its side effects) to better support highly concurrent environments
* fix to the type signature of `dds.keep` and `dds.eval`
* improves debugging messages (with a potential extra round trip to the store)

## v0.7.0

Adds a major feature: caching in memory of most recently used objects. See the documentation of
`dds.set_store`.

Other features:

* keyworded arguments are now accepted in `dds.keep` and `dds.eval`

## v0.6.0

This is a major release that changes
the algorithm of calculating the signatures.
*Upgrading from a previous version will trigger the cache to be calculated again*.

This change is not expected to happen again except for localized bug fixes.
