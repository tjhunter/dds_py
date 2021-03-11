# Changelog

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