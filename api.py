#
#
# @ks.describe(forget_args=["debug"])
# def my_function(f: pys.Dataset, debug = False) -> int:
#     return null
#
# res = ks.cache(my_function, f=null)
# res = ks.cache(my_function(f))
# res = ks.keep("/test/value", my_function(f))
# res = ks.keep("/test/value", my_function, f)
#
#
# res = ks.load("/test/value")
#
# model = ks.keep("/my_model", ks.aws.sagemaker(
#     aws.s3_dataset("s3://bucket/path"),
#     model=aws.sagemaker.pytorch({"num_epoch":3})
# ))
#
#
# def preprocess(train_df, other_df) -> df:
#     buckets = ks.keep(find_buckets(train_df))
#     return ks.ml.bucketize(buckets, other_df)
#
#
# def preprocess2(train_df):
#     buckets = find_buckets(train_df)
#     return lambda other_df: ks.ml.bucketize(buckets, other_df)
#
#
# @ks.function()
# def main():
#     df = sp.load("...")
#     (test_df, train_df) = ks.split(df)
#     preprocessor = ks.cache(preprocess2(train_df))
#     train_df = preprocessor(train_df)
#     model = aws.sagemaker(train_df)
#
#     test_df = preprocessor(test_df)
#     ks.ml.apply(model, test_df)
#
#
# def _loader():
#     return sp.load("/...")
#
# def load_data():
#     return ks.keep("/...", _loader)
#
#
# def transform(df):
#     df["x"] = 3
#     return df
#
# def etl():
#     df = load_data()
#     df = transform(df)
#     df2 = ks.load("/...")
#     return df.count() + df2.count()
#
#
# ks.eval(etl)
#
#
# def main2():
#     df = sp.load("...")
#     df = df["xx"]
#
#
#
#
# # *** model ***
#
# const = "" + arg()
#
#
# def f0():
#     x = ks.load("")
#     return x
#
# def f1(arg1):
#     x = ks.load("")
#     x = ks.load(const)
#     if arg1:
#         x = ks.load(const)
#     x = ks.load(arg1)  # forbidden? should be authorized
#     x = ks.load(arg1 + "")  # forbidden
#     x = ks.load(f3())  # forbidden?
#     x = f0()
#     return x
#
#
# def f2(arg2):
#     x = ks.keep(const, f1, arg1=arg2)
#     x = ks.keep("", f1, arg1=arg2)
#     return x
#
# f2(None)
# f2(None)
#
# ks.keep(const, x)
# f2(None)
#

"""
DEBUG:dds._api:locals: ['args', 'fun', 'kwargs', 'path']
DEBUG:dds.fun_args:get_arg_ctx: <function _related_party_info at 0x7f5d1031add0>: arg_sig=() -> pyspark.sql.dataframe.DataFrame args=()
DEBUG:dds._api:arg_ctx: FunctionArgContext(named_args=OrderedDict(), inner_call_key=None)
DEBUG:dds.introspect:Starting _introspect: <function _related_party_info at 0x7f5d1031add0>: arg_sig=() -> pyspark.sql.dataframe.DataFrame
DEBUG:dds.introspect:local vars: ['code', 'codes', 'foundations', 'party', 'party_to_party', 'related_parties', 'related_party_info', 'relationship_types_to_company']
DEBUG:dds.introspect:inspect_fun: local_vars: {'party_to_party', 'foundations', 'party', 'code', 'related_party_info', 'relationship_types_to_company', 'codes', 'related_parties'}
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: spark local_dep_path:spark
DEBUG:dds.introspect:_retrieve_object_rec: spark <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'>
DEBUG:dds.introspect:_retrieve_object_rec: <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'> is not definition module, redirecting to <module 'dsigi_ctf.spark_utils' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/spark_utils.py'>
DEBUG:dds.introspect:_retrieve_object_rec: spark <module 'dsigi_ctf.spark_utils' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/spark_utils.py'>
DEBUG:dds.introspect:is_authorized_path: {'dds', '__main__', 'dsigi_ctf'}
DEBUG:dds.introspect:is_authorized_path: {'dds', '__main__', 'dsigi_ctf'}
DEBUG:dds.introspect:Object spark (<class 'function'>) of path <dsigi_ctf.spark_utils.spark> is authorized,
DEBUG:dds.introspect:visit name spark: skipping (fun)
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: spark local_dep_path:spark
DEBUG:dds.introspect:_retrieve_object_rec: spark <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'>
DEBUG:dds.introspect:_retrieve_object_rec: <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'> is not definition module, redirecting to <module 'dsigi_ctf.spark_utils' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/spark_utils.py'>
DEBUG:dds.introspect:_retrieve_object_rec: spark <module 'dsigi_ctf.spark_utils' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/spark_utils.py'>
DEBUG:dds.introspect:is_authorized_path: {'dds', '__main__', 'dsigi_ctf'}
DEBUG:dds.introspect:is_authorized_path: {'dds', '__main__', 'dsigi_ctf'}
DEBUG:dds.introspect:Object spark (<class 'function'>) of path <dsigi_ctf.spark_utils.spark> is authorized,
DEBUG:dds.introspect:visit name spark: skipping (fun)
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: party_to_party local_dep_path:party_to_party
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: party_to_party skipping ctx: <_ast.Store object at 0x7f5d4d0d89d0>
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: spark local_dep_path:spark
DEBUG:dds.introspect:_retrieve_object_rec: spark <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'>
DEBUG:dds.introspect:_retrieve_object_rec: <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'> is not definition module, redirecting to <module 'dsigi_ctf.spark_utils' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/spark_utils.py'>
DEBUG:dds.introspect:_retrieve_object_rec: spark <module 'dsigi_ctf.spark_utils' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/spark_utils.py'>
DEBUG:dds.introspect:is_authorized_path: {'dds', '__main__', 'dsigi_ctf'}
DEBUG:dds.introspect:is_authorized_path: {'dds', '__main__', 'dsigi_ctf'}
DEBUG:dds.introspect:Object spark (<class 'function'>) of path <dsigi_ctf.spark_utils.spark> is authorized,
DEBUG:dds.introspect:visit name spark: skipping (fun)
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: path_source_party2party local_dep_path:path_source_party2party
DEBUG:dds.introspect:_retrieve_object_rec: path_source_party2party <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'>
DEBUG:dds.introspect:is_authorized_path: {'dds', '__main__', 'dsigi_ctf'}
DEBUG:dds.introspect:Object path_source_party2party (<class 'str'>) of path <dsigi_ctf.initial_tables.path_source_party2party> is authorized,
DEBUG:dds.introspect:Cache key: <dsigi_ctf.initial_tables.path_source_party2party>: <class 'str'> 84efbd92c2187b17939ee449d9b9d83f3e7d477dfc11125be0c2505372a7c7cd
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: foundations local_dep_path:foundations
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: foundations skipping ctx: <_ast.Store object at 0x7f5d4d0d89d0>
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: load_foundations local_dep_path:load_foundations
DEBUG:dds.introspect:_retrieve_object_rec: load_foundations <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'>
DEBUG:dds.introspect:is_authorized_path: {'dds', '__main__', 'dsigi_ctf'}
DEBUG:dds.introspect:is_authorized_path: {'dds', '__main__', 'dsigi_ctf'}
DEBUG:dds.introspect:Object load_foundations (<class 'function'>) of path <dsigi_ctf.initial_tables.load_foundations> is authorized,
DEBUG:dds.introspect:visit name load_foundations: skipping (fun)
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: codes local_dep_path:codes
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: codes skipping ctx: <_ast.Store object at 0x7f5d4d0d89d0>
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: list local_dep_path:list
DEBUG:dds.introspect:Could not find list in <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'>, attempting a direct load
DEBUG:dds.introspect:Could not load name list, looking into the globals
DEBUG:dds.introspect:list not found in start_globals
DEBUG:dds.introspect:visit_Name: list: skipping (unauthorized)
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: range local_dep_path:range
DEBUG:dds.introspect:Could not find range in <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'>, attempting a direct load
DEBUG:dds.introspect:Could not load name range, looking into the globals
DEBUG:dds.introspect:range not found in start_globals
DEBUG:dds.introspect:visit_Name: range: skipping (unauthorized)
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: relationship_types_to_company local_dep_path:relationship_types_to_company
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: relationship_types_to_company skipping ctx: <_ast.Store object at 0x7f5d4d0d89d0>
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: code local_dep_path:code
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: code skipping, in vars
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: code local_dep_path:code
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: code skipping ctx: <_ast.Store object at 0x7f5d4d0d89d0>
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: codes local_dep_path:codes
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: codes skipping, in vars
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: relationship_types_to_company local_dep_path:relationship_types_to_company
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: relationship_types_to_company skipping, in vars
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties local_dep_path:related_parties
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties skipping ctx: <_ast.Store object at 0x7f5d4d0d89d0>
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: party_to_party local_dep_path:party_to_party
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: party_to_party skipping, in vars
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: F local_dep_path:F
DEBUG:dds.introspect:_retrieve_object_rec: F <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'>
DEBUG:dds.introspect:_retrieve_object_rec: . <module 'pyspark.sql.functions' from '/databricks/spark/python/pyspark/sql/functions.py'>
DEBUG:dds.introspect:is_authorized_path: {'dds', '__main__', 'dsigi_ctf'}
DEBUG:dds.introspect:_retrieve_object_rec: Actual module <pyspark.sql.functions> for obj <module 'pyspark.sql.functions' from '/databricks/spark/python/pyspark/sql/functions.py'> is not authorized
DEBUG:dds.introspect:visit_Name: F: skipping (unauthorized)
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: relationship_types_to_company local_dep_path:relationship_types_to_company
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: relationship_types_to_company skipping, in vars
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties local_dep_path:related_parties
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties skipping ctx: <_ast.Store object at 0x7f5d4d0d89d0>
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: foundations local_dep_path:foundations
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: foundations skipping, in vars
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties local_dep_path:related_parties
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties skipping, in vars
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties local_dep_path:related_parties
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties skipping ctx: <_ast.Store object at 0x7f5d4d0d89d0>
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties local_dep_path:related_parties
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties skipping, in vars
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties local_dep_path:related_parties
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties skipping ctx: <_ast.Store object at 0x7f5d4d0d89d0>
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties local_dep_path:related_parties
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties skipping, in vars
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties local_dep_path:related_parties
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties skipping ctx: <_ast.Store object at 0x7f5d4d0d89d0>
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties local_dep_path:related_parties
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties skipping, in vars
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties local_dep_path:related_parties
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties skipping ctx: <_ast.Store object at 0x7f5d4d0d89d0>
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties local_dep_path:related_parties
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties skipping, in vars
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: party local_dep_path:party
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: party skipping ctx: <_ast.Store object at 0x7f5d4d0d89d0>
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: spark local_dep_path:spark
DEBUG:dds.introspect:_retrieve_object_rec: spark <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'>
DEBUG:dds.introspect:_retrieve_object_rec: <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'> is not definition module, redirecting to <module 'dsigi_ctf.spark_utils' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/spark_utils.py'>
DEBUG:dds.introspect:_retrieve_object_rec: spark <module 'dsigi_ctf.spark_utils' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/spark_utils.py'>
DEBUG:dds.introspect:is_authorized_path: {'dds', '__main__', 'dsigi_ctf'}
DEBUG:dds.introspect:is_authorized_path: {'dds', '__main__', 'dsigi_ctf'}
DEBUG:dds.introspect:Object spark (<class 'function'>) of path <dsigi_ctf.spark_utils.spark> is authorized,
DEBUG:dds.introspect:visit name spark: skipping (fun)
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: path_source_party local_dep_path:path_source_party
DEBUG:dds.introspect:_retrieve_object_rec: path_source_party <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'>
DEBUG:dds.introspect:is_authorized_path: {'dds', '__main__', 'dsigi_ctf'}
DEBUG:dds.introspect:Object path_source_party (<class 'str'>) of path <dsigi_ctf.initial_tables.path_source_party> is authorized,
DEBUG:dds.introspect:Cache key: <dsigi_ctf.initial_tables.path_source_party>: <class 'str'> 09fbfbd5542ba3b2b59dff79194251c75c6fe3dfd9ccc65b2c5dc77928bf3b23
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_party_info local_dep_path:related_party_info
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_party_info skipping ctx: <_ast.Store object at 0x7f5d4d0d89d0>
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties local_dep_path:related_parties
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_parties skipping, in vars
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: party local_dep_path:party
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: party skipping, in vars
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_party_info local_dep_path:related_party_info
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: related_party_info skipping, in vars
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: DataFrame local_dep_path:DataFrame
DEBUG:dds.introspect:_retrieve_object_rec: DataFrame <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'>
DEBUG:dds.introspect:_retrieve_object_rec: <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'> is not definition module, redirecting to <module 'pyspark.sql.dataframe' from '/databricks/spark/python/pyspark/sql/dataframe.py'>
DEBUG:dds.introspect:_retrieve_object_rec: DataFrame <module 'pyspark.sql.dataframe' from '/databricks/spark/python/pyspark/sql/dataframe.py'>
DEBUG:dds.introspect:is_authorized_path: {'dds', '__main__', 'dsigi_ctf'}
DEBUG:dds.introspect:Object DataFrame of type <class 'type'> and path <pyspark.sql.dataframe.DataFrame> is not authorized (path)
DEBUG:dds.introspect:visit_Name: DataFrame: skipping (unauthorized)
DEBUG:dds.introspect:inspect_fun: ext_deps: [ExternalDep(local_path=PurePosixPath('path_source_party'), path=<dsigi_ctf.initial_tables.path_source_party>, sig='09fbfbd5542ba3b2b59dff79194251c75c6fe3dfd9ccc65b2c5dc77928bf3b23'), ExternalDep(local_path=PurePosixPath('path_source_party2party'), path=<dsigi_ctf.initial_tables.path_source_party2party>, sig='84efbd92c2187b17939ee449d9b9d83f3e7d477dfc11125be0c2505372a7c7cd')]
DEBUG:dds.introspect:Inspect call:
 Call(
    lineno=2,
    col_offset=4,
    func=Attribute(
        lineno=2,
        col_offset=4,
        value=Call(
            lineno=2,
            col_offset=4,
            func=Name(lineno=2, col_offset=4, id='spark', ctx=Load()),
            args=[],
            keywords=[],
        ),
        attr='sql',
        ctx=Load(),
    ),
    args=[Str(lineno=2, col_offset=16, s='SET spark.sql.legacy.parquet.datetimeRebaseModeInRead=CORRECTED')],
    keywords=[],
)
DEBUG:dds.introspect:inspect_call: local_path: spark/sql
DEBUG:dds.introspect:_retrieve_object_rec: spark/sql <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'>
DEBUG:dds.introspect:Failed to consider object type <class 'function'> at path spark/sql context_mod: <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'>
DEBUG:dds.introspect:inspect_call: local_path: spark/sql is rejected
DEBUG:dds.introspect:Inspect call:
 Call(
    lineno=2,
    col_offset=4,
    func=Name(lineno=2, col_offset=4, id='spark', ctx=Load()),
    args=[],
    keywords=[],
)
DEBUG:dds.introspect:inspect_call: local_path: spark
DEBUG:dds.introspect:_retrieve_object_rec: spark <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'>
DEBUG:dds.introspect:_retrieve_object_rec: <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'> is not definition module, redirecting to <module 'dsigi_ctf.spark_utils' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/spark_utils.py'>
DEBUG:dds.introspect:_retrieve_object_rec: spark <module 'dsigi_ctf.spark_utils' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/spark_utils.py'>
DEBUG:dds.introspect:is_authorized_path: {'dds', '__main__', 'dsigi_ctf'}
DEBUG:dds.introspect:is_authorized_path: {'dds', '__main__', 'dsigi_ctf'}
DEBUG:dds.introspect:Object spark (<class 'function'>) of path <dsigi_ctf.spark_utils.spark> is authorized,
DEBUG:dds.introspect:Starting _introspect: <function spark at 0x7f5d1031a050>: arg_sig=() -> pyspark.sql.session.SparkSession
DEBUG:dds.introspect:local vars: []
DEBUG:dds.introspect:inspect_fun: local_vars: set()
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: SparkSession local_dep_path:SparkSession
DEBUG:dds.introspect:_retrieve_object_rec: SparkSession <module 'dsigi_ctf.spark_utils' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/spark_utils.py'>
DEBUG:dds.introspect:_retrieve_object_rec: <module 'dsigi_ctf.spark_utils' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/spark_utils.py'> is not definition module, redirecting to <module 'pyspark.sql.session' from '/databricks/spark/python/pyspark/sql/session.py'>
DEBUG:dds.introspect:_retrieve_object_rec: SparkSession <module 'pyspark.sql.session' from '/databricks/spark/python/pyspark/sql/session.py'>
DEBUG:dds.introspect:is_authorized_path: {'dds', '__main__', 'dsigi_ctf'}
DEBUG:dds.introspect:Object SparkSession of type <class 'type'> and path <pyspark.sql.session.SparkSession> is not authorized (path)
DEBUG:dds.introspect:visit_Name: SparkSession: skipping (unauthorized)
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: SparkSession local_dep_path:SparkSession
DEBUG:dds.introspect:_retrieve_object_rec: SparkSession <module 'dsigi_ctf.spark_utils' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/spark_utils.py'>
DEBUG:dds.introspect:_retrieve_object_rec: <module 'dsigi_ctf.spark_utils' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/spark_utils.py'> is not definition module, redirecting to <module 'pyspark.sql.session' from '/databricks/spark/python/pyspark/sql/session.py'>
DEBUG:dds.introspect:_retrieve_object_rec: SparkSession <module 'pyspark.sql.session' from '/databricks/spark/python/pyspark/sql/session.py'>
DEBUG:dds.introspect:is_authorized_path: {'dds', '__main__', 'dsigi_ctf'}
DEBUG:dds.introspect:Object SparkSession of type <class 'type'> and path <pyspark.sql.session.SparkSession> is not authorized (path)
DEBUG:dds.introspect:visit_Name: SparkSession: skipping (unauthorized)
DEBUG:dds.introspect:inspect_fun: ext_deps: []
DEBUG:dds.introspect:Inspect call:
 Call(
    lineno=2,
    col_offset=11,
    func=Attribute(
        lineno=2,
        col_offset=11,
        value=Name(lineno=2, col_offset=11, id='SparkSession', ctx=Load()),
        attr='getActiveSession',
        ctx=Load(),
    ),
    args=[],
    keywords=[],
)
DEBUG:dds.introspect:inspect_call: local_path: SparkSession/getActiveSession
DEBUG:dds.introspect:_retrieve_object_rec: SparkSession/getActiveSession <module 'dsigi_ctf.spark_utils' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/spark_utils.py'>
DEBUG:dds.introspect:Failed to consider object type <class 'type'> at path SparkSession/getActiveSession context_mod: <module 'dsigi_ctf.spark_utils' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/spark_utils.py'>
DEBUG:dds.introspect:inspect_call: local_path: SparkSession/getActiveSession is rejected
DEBUG:dds.introspect:inspect_fun: interactions: []
DEBUG:dds.introspect:End _introspect: <function spark at 0x7f5d1031a050>: FunctionInteractions(arg_input=FunctionArgContext(named_args=OrderedDict(), inner_call_key='4f318ea44d2e485418fa8c2834eb1add4f0feeb0957f4b70ac270364fbbf8906'), fun_body_sig='f70a6a8fa92873c0658d75812bdba40c1fcf1b36e48e7bc74167b4eb5a5fa02a', fun_return_sig='d558e185866dd2b914fdc1e2b1f38011ad4e869c2a8aa539fee5c4fcec49f9ea', external_deps=[], parsed_body=[], store_path=None, fun_path=<dsigi_ctf.spark_utils.spark>)
DEBUG:dds.introspect:Inspect call:
 Call(
    lineno=3,
    col_offset=4,
    func=Attribute(
        lineno=3,
        col_offset=4,
        value=Call(
            lineno=3,
            col_offset=4,
            func=Name(lineno=3, col_offset=4, id='spark', ctx=Load()),
            args=[],
            keywords=[],
        ),
        attr='sql',
        ctx=Load(),
    ),
    args=[Str(lineno=3, col_offset=16, s='SET spark.sql.legacy.parquet.datetimeRebaseModeInWrite=CORRECTED')],
    keywords=[],
)
DEBUG:dds.introspect:inspect_call: local_path: spark/sql
DEBUG:dds.introspect:_retrieve_object_rec: spark/sql <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'>
DEBUG:dds.introspect:Failed to consider object type <class 'function'> at path spark/sql context_mod: <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'>
DEBUG:dds.introspect:inspect_call: local_path: spark/sql is rejected
DEBUG:dds.introspect:Inspect call:
 Call(
    lineno=3,
    col_offset=4,
    func=Name(lineno=3, col_offset=4, id='spark', ctx=Load()),
    args=[],
    keywords=[],
)
DEBUG:dds.introspect:inspect_call: local_path: spark
DEBUG:dds.introspect:_retrieve_object_rec: spark <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'>
DEBUG:dds.introspect:_retrieve_object_rec: <module 'dsigi_ctf.initial_tables' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/initial_tables.py'> is not definition module, redirecting to <module 'dsigi_ctf.spark_utils' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/spark_utils.py'>
DEBUG:dds.introspect:_retrieve_object_rec: spark <module 'dsigi_ctf.spark_utils' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/spark_utils.py'>
DEBUG:dds.introspect:is_authorized_path: {'dds', '__main__', 'dsigi_ctf'}
DEBUG:dds.introspect:is_authorized_path: {'dds', '__main__', 'dsigi_ctf'}
DEBUG:dds.introspect:Object spark (<class 'function'>) of path <dsigi_ctf.spark_utils.spark> is authorized,
DEBUG:dds.introspect:Starting _introspect: <function spark at 0x7f5d1031a050>: arg_sig=() -> pyspark.sql.session.SparkSession
DEBUG:dds.introspect:local vars: []
DEBUG:dds.introspect:inspect_fun: local_vars: set()
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: SparkSession local_dep_path:SparkSession
DEBUG:dds.introspect:_retrieve_object_rec: SparkSession <module 'dsigi_ctf.spark_utils' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/spark_utils.py'>
DEBUG:dds.introspect:_retrieve_object_rec: <module 'dsigi_ctf.spark_utils' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/spark_utils.py'> is not definition module, redirecting to <module 'pyspark.sql.session' from '/databricks/spark/python/pyspark/sql/session.py'>
DEBUG:dds.introspect:_retrieve_object_rec: SparkSession <module 'pyspark.sql.session' from '/databricks/spark/python/pyspark/sql/session.py'>
DEBUG:dds.introspect:is_authorized_path: {'dds', '__main__', 'dsigi_ctf'}
DEBUG:dds.introspect:Object SparkSession of type <class 'type'> and path <pyspark.sql.session.SparkSession> is not authorized (path)
DEBUG:dds.introspect:visit_Name: SparkSession: skipping (unauthorized)
DEBUG:dds.introspect:ExternalVarsVisitor:visit_Name: id: SparkSession local_dep_path:SparkSession
DEBUG:dds.introspect:_retrieve_object_rec: SparkSession <module 'dsigi_ctf.spark_utils' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/spark_utils.py'>
DEBUG:dds.introspect:_retrieve_object_rec: <module 'dsigi_ctf.spark_utils' from '/databricks/python/lib/python3.7/site-packages/dsigi_ctf/spark_utils.py'> is not definition module, redirecting to <module 'pyspark.sql.session' from '/databricks/spark/python/pyspark/sql/session.py'>

*** WARNING: skipped 439843 bytes of output ***

INFO:dds._api:Interaction tree:
INFO:dds._api:`- Fun <dsigi_ctf.initial_tables._related_party_info> /mnt/dseedsi/dsigi_ctf/related_parties <- 617683685f4b80a64380532eae597734f3c65cf3596cc94c2c2136e68b17f2ae
INFO:dds._api:   |- Dep path_source_party -> <dsigi_ctf.initial_tables.path_source_party>: 09fbfbd5542ba3b2b59dff79194251c75c6fe3dfd9ccc65b2c5dc77928bf3b23
INFO:dds._api:   |- Dep path_source_party2party -> <dsigi_ctf.initial_tables.path_source_party2party>: 84efbd92c2187b17939ee449d9b9d83f3e7d477dfc11125be0c2505372a7c7cd
INFO:dds._api:   |- Fun <dsigi_ctf.spark_utils.spark> None <- d558e185866dd2b914fdc1e2b1f38011ad4e869c2a8aa539fee5c4fcec49f9ea
INFO:dds._api:   |  `- Ctx 4f318ea44d2e485418fa8c2834eb1add4f0feeb0957f4b70ac270364fbbf8906
INFO:dds._api:   |- Fun <dsigi_ctf.spark_utils.spark> None <- d558e185866dd2b914fdc1e2b1f38011ad4e869c2a8aa539fee5c4fcec49f9ea
INFO:dds._api:   |  `- Ctx 36d3a538972cfdc4e0c5ef1d45033ad529b64bc116ef1059e171a798ffdd6845
INFO:dds._api:   |- Fun <dsigi_ctf.spark_utils.spark> None <- d558e185866dd2b914fdc1e2b1f38011ad4e869c2a8aa539fee5c4fcec49f9ea
INFO:dds._api:   |  `- Ctx 306a0c4f11be4f6c99e1f9c07e3a30cee375ff6c9a988dbfe2fc45e3d4c709b8
INFO:dds._api:   |- Fun <dsigi_ctf.initial_tables.load_foundations> None <- 45cae7004abd3b55c5451d20cbbe0f8c0eee87119031c11e9c9cd383ae30b140
INFO:dds._api:   |  |- Ctx f0dc838be04dcf06d366bb6352c3a97a56860e30017953481407206a8782e069
INFO:dds._api:   |  |- Dep path_foundations -> <dsigi_ctf.initial_tables.path_foundations>: e3cf31230b32bacf67521bc155326b9142b86ac4485919889ee04f6ec4d977ad
INFO:dds._api:   |  `- Fun <dsigi_ctf.initial_tables._interesting_foundations> /mnt/dseedsi/dsigi_ctf/foundations <- dfde11c84d0c637f86c379c585000f06d06c72b6076d8935be0048f1db4bc812
INFO:dds._api:   |     |- Ctx eb7b3a1235a962c087c3d7d08fdea5b0eeac83702baf8cb758b08b5f6b2a9d6f
INFO:dds._api:   |     |- Dep path_source_product_contract -> <dsigi_ctf.initial_tables.path_source_product_contract>: 3889b279442114bc06de6fe87c7f2628caac01141bfc2f1bb90a05950ff14292
INFO:dds._api:   |     |- Fun <dsigi_ctf.spark_utils.spark> None <- d558e185866dd2b914fdc1e2b1f38011ad4e869c2a8aa539fee5c4fcec49f9ea
INFO:dds._api:   |     |  `- Ctx e0012c1ba6b6868d8e6a113be30bf80fac7716cbc670786d96dbb6e1a4bedeb7
INFO:dds._api:   |     |- Fun <dsigi_ctf.spark_utils.spark> None <- d558e185866dd2b914fdc1e2b1f38011ad4e869c2a8aa539fee5c4fcec49f9ea
INFO:dds._api:   |     |  `- Ctx ff9359eb17fb98cf6e6c9786eaf5ac4a2d9314b4890c456f92a099debfdf7987
INFO:dds._api:   |     |- Fun <dsigi_ctf.initial_tables._select_foundations> None <- d7a6209e09df028071089f6d25e1dc95604466e3f4ecab8a9f5125b9fd0ddfe9
INFO:dds._api:   |     |  |- Ctx 8c1ca5013b434786381616135032815719365c73a61f13b92d036e858e8ffed2
INFO:dds._api:   |     |  |- Dep path_source_apc_role -> <dsigi_ctf.initial_tables.path_source_apc_role>: 4c6b5d0ac7387ffdb8b27b39de1c70c569a1bd23d08e7b0f4b05390f322adbf8
INFO:dds._api:   |     |  |- Dep path_source_party -> <dsigi_ctf.initial_tables.path_source_party>: 09fbfbd5542ba3b2b59dff79194251c75c6fe3dfd9ccc65b2c5dc77928bf3b23
INFO:dds._api:   |     |  |- Dep path_source_party_role -> <dsigi_ctf.initial_tables.path_source_party_role>: 3f31ab85cac6988311e5dcd86e09640b3027d6e8bf350047cf9179fab3e63357
INFO:dds._api:   |     |  |- Fun <dsigi_ctf.spark_utils.spark> None <- d558e185866dd2b914fdc1e2b1f38011ad4e869c2a8aa539fee5c4fcec49f9ea
INFO:dds._api:   |     |  |  `- Ctx bbb1b376dc1761ed37a97e8561c48267ff0bef12926aef65d95e27e6373a6909
INFO:dds._api:   |     |  |- Fun <dsigi_ctf.spark_utils.spark> None <- d558e185866dd2b914fdc1e2b1f38011ad4e869c2a8aa539fee5c4fcec49f9ea
INFO:dds._api:   |     |  |  `- Ctx 12d710dd33e3b5aabd89dd300050a5fe846be5d2698228739090dd8c1f338402
INFO:dds._api:   |     |  |- Fun <dsigi_ctf.spark_utils.spark> None <- d558e185866dd2b914fdc1e2b1f38011ad4e869c2a8aa539fee5c4fcec49f9ea
INFO:dds._api:   |     |  |  `- Ctx ca3873262fa888d8a874098a1ffbc20225717739e73904b961fa8ef4fb47edb8
INFO:dds._api:   |     |  `- Fun <dsigi_ctf.spark_utils.spark> None <- d558e185866dd2b914fdc1e2b1f38011ad4e869c2a8aa539fee5c4fcec49f9ea
INFO:dds._api:   |     |     `- Ctx cf2ff8b737740d78f623c80a4bac90fa7fa2d824a8ded6ad12ba0cb79fa0eb32
INFO:dds._api:   |     |- Fun <dsigi_ctf.curated_sources.cash_wire> None <- ffc2e806fe76942e4c600912a4e0f3af9f8a71cdb6b29f5203796c88b6e0e0d9
INFO:dds._api:   |     |  |- Ctx 4a18599df4cdc0658fe65a7a7ce45e08f2317d81b2173ed9b079cf026893aa3c
INFO:dds._api:   |     |  |- Dep end_month -> <dsigi_ctf.curated_sources.end_month>: 6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b
INFO:dds._api:   |     |  |- Dep end_year -> <dsigi_ctf.curated_sources.end_year>: 73a2af8864fc500fa49048bf3003776c19938f360e56bd03663866fb3087884a
INFO:dds._api:   |     |  |- Dep path_source_cash_fccm -> <dsigi_ctf.curated_sources.path_source_cash_fccm>: b4ce80c1dc9940ada44ff6227331b453d2e496f8ab872211b0ace1b5dab5f0cb
INFO:dds._api:   |     |  |- Dep path_source_wire_fccm -> <dsigi_ctf.curated_sources.path_source_wire_fccm>: 443df96372ebf1964713969432548e5f29eb9c430ef93aa8e82787d5c62867bc
INFO:dds._api:   |     |  |- Dep start_month -> <dsigi_ctf.curated_sources.start_month>: d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35
INFO:dds._api:   |     |  |- Dep start_year -> <dsigi_ctf.curated_sources.start_year>: 152e69cf3c8e76c8d8b0aed924ddd1708e4c68624611af33d52c2c2814dd5df9
INFO:dds._api:   |     |  |- Fun <dsigi_ctf.spark_utils.spark> None <- d558e185866dd2b914fdc1e2b1f38011ad4e869c2a8aa539fee5c4fcec49f9ea
INFO:dds._api:   |     |  |  `- Ctx 50c350e1607cf6df7b34619febc03506735f0667074538d3c89f48e17fccaee1
INFO:dds._api:   |     |  |- Fun <dsigi_ctf.spark_utils.spark> None <- d558e185866dd2b914fdc1e2b1f38011ad4e869c2a8aa539fee5c4fcec49f9ea
INFO:dds._api:   |     |  |  `- Ctx 2d79880260e855437501a6a83c26eddb4fa102e9ce26d43047ef4adfcabb7e50
INFO:dds._api:   |     |  |- Fun <dsigi_ctf.spark_utils.spark> None <- d558e185866dd2b914fdc1e2b1f38011ad4e869c2a8aa539fee5c4fcec49f9ea
INFO:dds._api:   |     |  |  `- Ctx da94d318e2764ebd81d7023c77141d3cc8dcfd2986123f1dbf9cb5acfef642df
INFO:dds._api:   |     |  |- Fun <dsigi_ctf.spark_utils.spark> None <- d558e185866dd2b914fdc1e2b1f38011ad4e869c2a8aa539fee5c4fcec49f9ea
INFO:dds._api:   |     |  |  `- Ctx ac890aeacacf2839293a822a6e62cc57eb70da4b62bf9b5c331b0625d202c5e9
INFO:dds._api:   |     |  |- Fun <dsigi_ctf.spark_utils.spark> None <- d558e185866dd2b914fdc1e2b1f38011ad4e869c2a8aa539fee5c4fcec49f9ea
INFO:dds._api:   |     |  |  `- Ctx e1293d8777e02fd9bd4cc3b7b97128f39b41228d92176ce4671b0f0f452f7be6
INFO:dds._api:   |     |  `- Fun <dsigi_ctf.spark_utils.spark> None <- d558e185866dd2b914fdc1e2b1f38011ad4e869c2a8aa539fee5c4fcec49f9ea
INFO:dds._api:   |     |     `- Ctx 9c6bc96ffdaeaf3e1c0cdf5d77f185a98f43e81693ab1c23cfb10ab54e5519ab
INFO:dds._api:   |     |- Fun <dsigi_ctf.curated_sources.fccm_transactions> None <- 5b7c8746a1fa912bccccb2aba9979b8def411f01c80d0148c35c8d8d652f91c6
INFO:dds._api:   |     |  `- Ctx 43fe069f845aa37190344be064699310e24f2385b90cadd75a088d0242630399
INFO:dds._api:   |     |- Fun <dsigi_ctf.spark_utils.spark> None <- d558e185866dd2b914fdc1e2b1f38011ad4e869c2a8aa539fee5c4fcec49f9ea
INFO:dds._api:   |     |  `- Ctx ab4c23e52390eef6572322160959277235945809c8b6d3e1b36559e45448dafd
INFO:dds._api:   |     `- Fun <dsigi_ctf.curated_sources.fccm_apcid> None <- b55af0d39290165716ff7a727811cdc4451489f4f90874bc6b56eabcedba2161
INFO:dds._api:   |        |- Ctx 5f5a096f4ef05656f49016baaf4572cc5f25ef14b1e02f1afaf08725d58b87df
INFO:dds._api:   |        `- Fun <dsigi_ctf.curated_sources._prefix_acct> None <- 928bef6e289fab82b02e5ad49c15b43a0a6da075839f2cfd4099e5f6d0a53d62
INFO:dds._api:   |           `- Ctx a2f728cc12ac33c0ceab266b1358d10f76ae7f330518266ce5d08510a9f7cd45
INFO:dds._api:   `- Fun <dsigi_ctf.spark_utils.spark> None <- d558e185866dd2b914fdc1e2b1f38011ad4e869c2a8aa539fee5c4fcec49f9ea
INFO:dds._api:      `- Ctx c79e5ef4d33dbbf65d00054c1962296b1e496491d5787d7d6e88f3e8ab20ecd3
DEBUG:dds._api:_eval_new_ctx:current_sig: 617683685f4b80a64380532eae597734f3c65cf3596cc94c2c2136e68b17f2ae
DEBUG:dds.codecs.databricks:Attempting to read metadata for key 617683685f4b80a64380532eae597734f3c65cf3596cc94c2c2136e68b17f2ae: /DSI_GI_CTF/data_cache/blobs/617683685f4b80a64380532eae597734f3c65cf3596cc94c2c2136e68b17f2ae.meta
DEBUG:dds.codecs.databricks:Could not read metadata for key 617683685f4b80a64380532eae597734f3c65cf3596cc94c2c2136e68b17f2ae: An error occurred while calling z:com.databricks.backend.daemon.dbutils.FSUtils.head.
: java.io.FileNotFoundException: /DSI_GI_CTF/data_cache/blobs/617683685f4b80a64380532eae597734f3c65cf3596cc94c2c2136e68b17f2ae.meta
	at com.databricks.backend.daemon.data.client.DatabricksFileSystemV2.$anonfun$getFileStatus$2(DatabricksFileSystemV2.scala:774)
	at com.databricks.s3a.S3AExeceptionUtils$.convertAWSExceptionToJavaIOException(DatabricksStreamUtils.scala:66)
	at com.databricks.backend.daemon.data.client.DatabricksFileSystemV2.$anonfun$getFileStatus$1(DatabricksFileSystemV2.scala:760)
	at com.databricks.logging.UsageLogging.$anonfun$recordOperation$4(UsageLogging.scala:429)
	at com.databricks.logging.UsageLogging.$anonfun$withAttributionContext$1(UsageLogging.scala:237)
	at scala.util.DynamicVariable.withValue(DynamicVariable.scala:62)
	at com.databricks.logging.UsageLogging.withAttributionContext(UsageLogging.scala:232)
	at com.databricks.logging.UsageLogging.withAttributionContext$(UsageLogging.scala:229)
	at com.databricks.backend.daemon.data.client.DatabricksFileSystemV2.withAttributionContext(DatabricksFileSystemV2.scala:454)
	at com.databricks.logging.UsageLogging.withAttributionTags(UsageLogging.scala:274)
	at com.databricks.logging.UsageLogging.withAttributionTags$(UsageLogging.scala:267)
	at com.databricks.backend.daemon.data.client.DatabricksFileSystemV2.withAttributionTags(DatabricksFileSystemV2.scala:454)
	at com.databricks.logging.UsageLogging.recordOperation(UsageLogging.scala:410)
	at com.databricks.logging.UsageLogging.recordOperation$(UsageLogging.scala:336)
	at com.databricks.backend.daemon.data.client.DatabricksFileSystemV2.recordOperation(DatabricksFileSystemV2.scala:454)
	at com.databricks.backend.daemon.data.client.DatabricksFileSystemV2.getFileStatus(DatabricksFileSystemV2.scala:760)
	at com.databricks.backend.daemon.data.client.DatabricksFileSystem.getFileStatus(DatabricksFileSystem.scala:201)
	at com.databricks.backend.daemon.dbutils.FSUtils$.$anonfun$head$1(DBUtilsCore.scala:192)
	at com.databricks.backend.daemon.dbutils.FSUtils$.withFsSafetyCheck(DBUtilsCore.scala:81)
	at com.databricks.backend.daemon.dbutils.FSUtils$.head(DBUtilsCore.scala:190)
	at com.databricks.backend.daemon.dbutils.FSUtils.head(DBUtilsCore.scala)
	at sun.reflect.GeneratedMethodAccessor520.invoke(Unknown Source)
	at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
	at java.lang.reflect.Method.invoke(Method.java:498)
	at py4j.reflection.MethodInvoker.invoke(MethodInvoker.java:244)
	at py4j.reflection.ReflectionEngine.invoke(ReflectionEngine.java:380)
	at py4j.Gateway.invoke(Gateway.java:295)
	at py4j.commands.AbstractCommand.invokeMethod(AbstractCommand.java:132)
	at py4j.commands.CallCommand.execute(CallCommand.java:79)
	at py4j.GatewayConnection.run(GatewayConnection.java:251)
	at java.lang.Thread.run(Thread.java:748)

DEBUG:dds._api:Updating path: /mnt/dseedsi/dsigi_ctf/related_parties -> 617683685f4b80a64380532eae597734f3c65cf3596cc94c2c2136e68b17f2ae
DEBUG:dds._api:Updating path: /mnt/dseedsi/dsigi_ctf/foundations -> dfde11c84d0c637f86c379c585000f06d06c72b6076d8935be0048f1db4bc812
INFO:dds._api:_eval_new_ctx:Evaluating (eval) fun <function _related_party_info at 0x7f5d1031add0> with args [] kwargs OrderedDict()
DEBUG:dds.codecs.databricks:Attempting to read metadata for key dfde11c84d0c637f86c379c585000f06d06c72b6076d8935be0048f1db4bc812: /DSI_GI_CTF/data_cache/blobs/dfde11c84d0c637f86c379c585000f06d06c72b6076d8935be0048f1db4bc812.meta
DEBUG:dds.codecs.databricks:Could not read metadata for key dfde11c84d0c637f86c379c585000f06d06c72b6076d8935be0048f1db4bc812: An error occurred while calling z:com.databricks.backend.daemon.dbutils.FSUtils.head.
: java.io.FileNotFoundException: /DSI_GI_CTF/data_cache/blobs/dfde11c84d0c637f86c379c585000f06d06c72b6076d8935be0048f1db4bc812.meta
	at com.databricks.backend.daemon.data.client.DatabricksFileSystemV2.$anonfun$getFileStatus$2(DatabricksFileSystemV2.scala:774)
	at com.databricks.s3a.S3AExeceptionUtils$.convertAWSExceptionToJavaIOException(DatabricksStreamUtils.scala:66)
	at com.databricks.backend.daemon.data.client.DatabricksFileSystemV2.$anonfun$getFileStatus$1(DatabricksFileSystemV2.scala:760)
	at com.databricks.logging.UsageLogging.$anonfun$recordOperation$4(UsageLogging.scala:429)
	at com.databricks.logging.UsageLogging.$anonfun$withAttributionContext$1(UsageLogging.scala:237)
	at scala.util.DynamicVariable.withValue(DynamicVariable.scala:62)
	at com.databricks.logging.UsageLogging.withAttributionContext(UsageLogging.scala:232)
	at com.databricks.logging.UsageLogging.withAttributionContext$(UsageLogging.scala:229)
	at com.databricks.backend.daemon.data.client.DatabricksFileSystemV2.withAttributionContext(DatabricksFileSystemV2.scala:454)
	at com.databricks.logging.UsageLogging.withAttributionTags(UsageLogging.scala:274)
	at com.databricks.logging.UsageLogging.withAttributionTags$(UsageLogging.scala:267)
	at com.databricks.backend.daemon.data.client.DatabricksFileSystemV2.withAttributionTags(DatabricksFileSystemV2.scala:454)
	at com.databricks.logging.UsageLogging.recordOperation(UsageLogging.scala:410)
	at com.databricks.logging.UsageLogging.recordOperation$(UsageLogging.scala:336)
	at com.databricks.backend.daemon.data.client.DatabricksFileSystemV2.recordOperation(DatabricksFileSystemV2.scala:454)
	at com.databricks.backend.daemon.data.client.DatabricksFileSystemV2.getFileStatus(DatabricksFileSystemV2.scala:760)
	at com.databricks.backend.daemon.data.client.DatabricksFileSystem.getFileStatus(DatabricksFileSystem.scala:201)
	at com.databricks.backend.daemon.dbutils.FSUtils$.$anonfun$head$1(DBUtilsCore.scala:192)
	at com.databricks.backend.daemon.dbutils.FSUtils$.withFsSafetyCheck(DBUtilsCore.scala:81)
	at com.databricks.backend.daemon.dbutils.FSUtils$.head(DBUtilsCore.scala:190)
	at com.databricks.backend.daemon.dbutils.FSUtils.head(DBUtilsCore.scala)
	at sun.reflect.GeneratedMethodAccessor520.invoke(Unknown Source)
	at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
	at java.lang.reflect.Method.invoke(Method.java:498)
	at py4j.reflection.MethodInvoker.invoke(MethodInvoker.java:244)
	at py4j.reflection.ReflectionEngine.invoke(ReflectionEngine.java:380)
	at py4j.Gateway.invoke(Gateway.java:295)
	at py4j.commands.AbstractCommand.invokeMethod(AbstractCommand.java:132)
	at py4j.commands.CallCommand.execute(CallCommand.java:79)
	at py4j.GatewayConnection.run(GatewayConnection.java:251)
	at java.lang.Thread.run(Thread.java:748)
_interesting_foundations
INFO:dds._api:_eval:Evaluating (keep:/mnt/dseedsi/dsigi_ctf/foundations) fun <function _interesting_foundations at 0x7f5d1031aa70> with args [] kwargs OrderedDict()
INFO:dds._api:_eval:Evaluating (keep:/mnt/dseedsi/dsigi_ctf/foundations) fun <function _interesting_foundations at 0x7f5d1031aa70>: completed
INFO:dds._api:_eval:Storing blob into key dfde11c84d0c637f86c379c585000f06d06c72b6076d8935be0048f1db4bc812
DEBUG:dds.codecs.databricks:Committed dataframe to parquet: /DSI_GI_CTF/data_cache/blobs/dfde11c84d0c637f86c379c585000f06d06c72b6076d8935be0048f1db4bc812
Wrote 31 bytes.
DEBUG:dds.codecs.databricks:Committed new blob in dfde11c84d0c637f86c379c585000f06d06c72b6076d8935be0048f1db4bc812
INFO:dds._api:_eval_new_ctx:Evaluating (eval) fun <function _related_party_info at 0x7f5d1031add0>: completed
INFO:dds._api:_eval:Storing blob into key 617683685f4b80a64380532eae597734f3c65cf3596cc94c2c2136e68b17f2ae
DEBUG:dds.codecs.databricks:Committed dataframe to parquet: /DSI_GI_CTF/data_cache/blobs/617683685f4b80a64380532eae597734f3c65cf3596cc94c2c2136e68b17f2ae
Wrote 31 bytes.
DEBUG:dds.codecs.databricks:Committed new blob in 617683685f4b80a64380532eae597734f3c65cf3596cc94c2c2136e68b17f2ae
DEBUG:dds.codecs.databricks:Attempting to read metadata for key 617683685f4b80a64380532eae597734f3c65cf3596cc94c2c2136e68b17f2ae: /DSI_GI_CTF/data_managed/_dds_meta/mnt/dseedsi/dsigi_ctf/related_parties _dds_meta/mnt/dseedsi/dsigi_ctf/related_parties /mnt/dseedsi/dsigi_ctf/related_parties
DEBUG:dds.codecs.databricks:Path /mnt/dseedsi/dsigi_ctf/related_parties needs update (registered key 91fdcdde17e61a0c104c15720b7ed2776c16717481a9b6275e8a5d8b0f7630c7 != 617683685f4b80a64380532eae597734f3c65cf3596cc94c2c2136e68b17f2ae)
DEBUG:dds.codecs.databricks:Copying /DSI_GI_CTF/data_cache/blobs/617683685f4b80a64380532eae597734f3c65cf3596cc94c2c2136e68b17f2ae -> /DSI_GI_CTF/data_managed/mnt/dseedsi/dsigi_ctf/related_parties
DEBUG:dds.codecs.databricks:Linking new file /DSI_GI_CTF/data_managed/mnt/dseedsi/dsigi_ctf/related_parties
Wrote 87 bytes.
DEBUG:dds.codecs.databricks:Attempting to read metadata for key dfde11c84d0c637f86c379c585000f06d06c72b6076d8935be0048f1db4bc812: /DSI_GI_CTF/data_managed/_dds_meta/mnt/dseedsi/dsigi_ctf/foundations _dds_meta/mnt/dseedsi/dsigi_ctf/foundations /mnt/dseedsi/dsigi_ctf/foundations
DEBUG:dds.codecs.databricks:Path /mnt/dseedsi/dsigi_ctf/foundations needs update (registered key b4fe214a96951470be1ddece8164a95fa471e8dbf9193256d08739edaac1b154 != dfde11c84d0c637f86c379c585000f06d06c72b6076d8935be0048f1db4bc812)
DEBUG:dds.codecs.databricks:Copying /DSI_GI_CTF/data_cache/blobs/dfde11c84d0c637f86c379c585000f06d06c72b6076d8935be0048f1db4bc812 -> /DSI_GI_CTF/data_managed/mnt/dseedsi/dsigi_ctf/foundations
DEBUG:dds.codecs.databricks:Linking new file /DSI_GI_CTF/data_managed/mnt/dseedsi/dsigi_ctf/foundations
Wrote 87 bytes.
"""