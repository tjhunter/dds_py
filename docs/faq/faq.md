# Frequently asked questions

__What can’t you do with pure docker that you would need DDS for? 
e.g. Docker does a lot of caching of layers__


Docker fills a gap that slightly overlaps with DDS:

- Docker allows you to embed arbitrary content (software, models, data, ...) into a single bundle
- It requires a specific language (the Dockerfile instructions)
- Its caching system does not understand the semantics of your code: if you just move code, it will rebuild the layer. In fact, Docker has multiple caching systems that try to address this issue in different ways.
- It requires a specific runtime (the docker system) to run

By contrast, DDS understands very well your python code, and only your python code:

- you can run it in any python environment
- it will understand your code: if you copy/paste functions in files, 
it will understand that the code is still the same and will not trigger recomputations

In practice, both systems are complementary:

- you build all your data artifacts (models, cleaned data) with dds
- you embed them in a final docker container that you publish as a service, with MLFlow for example


__Can DDS run in the background automatically like Delta IO?__

Not currently, but this is a potential point on the roadmap. DDS already benefits from 
Delta IO if available, and solves a different problem:
 - DDS helps for batch transforms written in Python
 - Delta IO can be used for streaming and batch, using Python, Java
 - DDS automatically infers all the data dependencies from the code
 - Delta IO needs an explicit computation graph provided by the user

__Best practices: at which point in code should put it in?__

The rule of thumb is the following: any idempotent calculation that you end up waiting 
for and that takes more than 0.3 seconds to compute can benefit from DDS.

In practice, this includes:

 - fetching data from the internet and returning a `pandas` dataframe
 - using the `display()` function to show statistics on large tables
 - running ML models

With DDS, the general user experience is that any notebook can be made to run 
in less than 10 seconds. This is very powerful to communicate results 
that potentially depend on long-running calculations.
