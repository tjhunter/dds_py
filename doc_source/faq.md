# Frequently asked questions

__What can’t you do with pure docker that you would need DDS for? 
e.g. Docker does a lot of caching of layers__


Docker fills a gap that slightly overlaps indeed:

- Docker allows you to embed arbitrary content (software, models, data, ...) into a single bundle
- It requires a specific language (the Dockerfile instructions)
- Its caching system does not understand the semantics of your code: if you just move code, it will rebuild the layer. In fact, Docker has multiple caching systems that try to address this issue in different ways.
- It requires a specific runtime (the docker system) to run

By contrast, DDS understands very well your python code, and only your python code:

- you can run it in any python environment
- it will understand (to some extent) your code: if you copy/paste functions in files, it will understand that the code is still the same and will not trigger recomputations

In practice, both systems are complementary:
- you build all your data artifacts (models, cleaned data) with dds
- you embed them in a final docker container that you publish as a service, with MLFlow for example


* Can this run in the background automatically like delta io?

It does not currently do it, but it could. Actually, it is even better in theory than delta io because it understands all the dependencies between the functions and it can automatically parallelizes all the computations. The user does not even have to put any python parallel code.

* Best practices: at which point in code should put it in?

The rule of thumb that we use: anything that takes more than a minute to compute 
is tagged. That also includes the display() functions in databricks when showing 
stats. Our expectation now is that when the cache is filled, *any* notebook 
in databricks takes < 10 seconds to execute, whatever the size of the data or 
the duration of model training.

This is one of the best parts: you start in your notebook.
 You debug and write all your tests over there. Then you copy/paste your 
 code in a repo in which you have good engineering practices. If the code 
has not changed, the shared cache will pick up the content of the cache already 
populated, nothing will get rerun.

