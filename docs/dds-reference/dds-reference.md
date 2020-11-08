# DDS package documentation

## Main functions

**dds.keep(
    path: Union[str, DDSPath, pathlib.Path],
    fun: Callable[..., _Out],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> _Out:** 

::: dds.keep

**dds.eval**

```python
def eval(
    fun: Callable[..., _Out],
    *args: Tuple[Any, ...],
    dds_export_graph: Union[str, pathlib.Path, None] = None,
    dds_extra_debug: Optional[bool] = None,
    dds_stages: Optional[List[Union[str, ProcessingStage]]] = None,
    **kwargs: Dict[str, Any]
) -> Optional[_Out]
```

::: dds.eval

**dds.set_store**

::: dds.set_store

**dds.whitelist_module**

::: dds.whitelist_module

## Databricks integration

TODO: detail DBFS integration

TODO: dds.codecs.databricks.displayGraph
