# Changelog

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
