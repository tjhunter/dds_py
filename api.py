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

DEBUG    dds._api:_api.py:146 _eval_new_ctx:current_sig: fc05cffdbcf25e9da8612976b2b072347eeed616492a7e8a7e9f8ed8604eb0a7
DEBUG    dds._api:_api.py:148 _eval_new_ctx:Return cached signature fc05cffdbcf25e9da8612976b2b072347eeed616492a7e8a7e9f8ed8604eb0a7
DEBUG    dds._api:_api.py:132 locals: ['args', 'fun', 'kwargs', 'path']
DEBUG    dds.fun_args:fun_args.py:53 get_arg_ctx: <function _fun at 0x7f6ea22421f0>: arg_sig=(x=True) args=(False,)
DEBUG    dds.fun_args:fun_args.py:57 get_arg_ctx: <function _fun at 0x7f6ea22421f0>: idx=0 n=x p=x=True
DEBUG    dds._api:_api.py:135 arg_ctx: FunctionArgContext(named_args=OrderedDict([('x', '60a33e6cf5151f2d52eddae9685cfa270426aa89d8dbc7dfb854606f1d1a40fe')]), inner_call_key=None)
DEBUG    dds.introspect:introspect.py:93 Starting _introspect: <function _fun at 0x7f6ea22421f0>: arg_sig=(x=True)
DEBUG    dds.introspect:introspect.py:297 local vars: ['x']
DEBUG    dds.introspect:introspect.py:323 inspect_fun: local_vars: {'x'}
DEBUG    dds.introspect:introspect.py:169 ExternalVarsVisitor:visit_Name: id: _c local_dep_path:_c
DEBUG    dds.introspect:introspect.py:498 _retrieve_object_rec: _c <module 'dds_tests.test_default_args_basic' from '/mnt/c/Users/C76581/work/dds_py/dds_tests/test_default_args_basic.py'>
DEBUG    dds.introspect:introspect.py:78 is_authorized_path: {'dds_tests.test_default_args_basic', 'dds_tests.test_cache_nested', 'dds_tests.test_method_as_var', 'dds_tests.test_sklearn', 'dds_tests.test_refs', 'dds', 'dds_tests.test_repeat_nested', '__main__', 'dds_tests.test_basic', 'dds_tests.test_keep_direct_call', 'dds_tests.test_pathlib_basic'}
DEBUG    dds.introspect:introspect.py:78 is_authorized_path: {'dds_tests.test_default_args_basic', 'dds_tests.test_cache_nested', 'dds_tests.test_method_as_var', 'dds_tests.test_sklearn', 'dds_tests.test_refs', 'dds', 'dds_tests.test_repeat_nested', '__main__', 'dds_tests.test_basic', 'dds_tests.test_keep_direct_call', 'dds_tests.test_pathlib_basic'}
DEBUG    dds.introspect:introspect.py:554 Object _c of type <class 'dds_tests.utils.Counter'> is not authorized (type), dropping path <dds_tests.test_default_args_basic._c>
DEBUG    dds.introspect:introspect.py:570 Failed to consider object type <class 'dds_tests.utils.Counter'> at path _c context_mod: <module 'dds_tests.test_default_args_basic' from '/mnt/c/Users/C76581/work/dds_py/dds_tests/test_default_args_basic.py'>
DEBUG    dds.introspect:introspect.py:199 visit_Name: _c: skipping (unauthorized)
DEBUG    dds.introspect:introspect.py:169 ExternalVarsVisitor:visit_Name: id: str local_dep_path:str
DEBUG    dds.introspect:introspect.py:452 Could not find str in <module 'dds_tests.test_default_args_basic' from '/mnt/c/Users/C76581/work/dds_py/dds_tests/test_default_args_basic.py'>, attempting a direct load
DEBUG    dds.introspect:introspect.py:460 Could not load name str, looking into the globals
DEBUG    dds.introspect:introspect.py:487 str not found in start_globals
DEBUG    dds.introspect:introspect.py:199 visit_Name: str: skipping (unauthorized)
DEBUG    dds.introspect:introspect.py:169 ExternalVarsVisitor:visit_Name: id: x local_dep_path:x
DEBUG    dds.introspect:introspect.py:181 ExternalVarsVisitor:visit_Name: id: x skipping, in vars
DEBUG    dds.introspect:introspect.py:327 inspect_fun: ext_deps: []
DEBUG    dds.introspect:introspect.py:359 Inspect call:
 Call(
    lineno=2,
    col_offset=4,
    end_lineno=2,
    end_col_offset=18,
    func=Attribute(
        lineno=2,
        col_offset=4,
        end_lineno=2,
        end_col_offset=16,
        value=Name(lineno=2, col_offset=4, end_lineno=2, end_col_offset=6, id='_c', ctx=Load()),
        attr='increment',
        ctx=Load(),
    ),
    args=[],
    keywords=[],
)
DEBUG    dds.introspect:introspect.py:362 inspect_call: local_path: _c/increment
DEBUG    dds.introspect:introspect.py:498 _retrieve_object_rec: _c/increment <module 'dds_tests.test_default_args_basic' from '/mnt/c/Users/C76581/work/dds_py/dds_tests/test_default_args_basic.py'>
DEBUG    dds.introspect:introspect.py:570 Failed to consider object type <class 'dds_tests.utils.Counter'> at path _c/increment context_mod: <module 'dds_tests.test_default_args_basic' from '/mnt/c/Users/C76581/work/dds_py/dds_tests/test_default_args_basic.py'>
DEBUG    dds.introspect:introspect.py:370 inspect_call: local_path: _c/increment is rejected
DEBUG    dds.introspect:introspect.py:359 Inspect call:
 Call(
    lineno=3,
    col_offset=17,
    end_lineno=3,
    end_col_offset=23,
    func=Name(lineno=3, col_offset=17, end_lineno=3, end_col_offset=20, id='str', ctx=Load()),
    args=[Name(lineno=3, col_offset=21, end_lineno=3, end_col_offset=22, id='x', ctx=Load())],
    keywords=[],
)
DEBUG    dds.introspect:introspect.py:362 inspect_call: local_path: str
DEBUG    dds.introspect:introspect.py:452 Could not find str in <module 'dds_tests.test_default_args_basic' from '/mnt/c/Users/C76581/work/dds_py/dds_tests/test_default_args_basic.py'>, attempting a direct load
DEBUG    dds.introspect:introspect.py:460 Could not load name str, looking into the globals
DEBUG    dds.introspect:introspect.py:487 str not found in start_globals
DEBUG    dds.introspect:introspect.py:370 inspect_call: local_path: str is rejected
DEBUG    dds.introspect:introspect.py:334 inspect_fun: interactions: []
DEBUG    dds.introspect:introspect.py:111 End _introspect: <function _fun at 0x7f6ea22421f0>: FunctionInteractions(arg_input=FunctionArgContext(named_args=OrderedDict([('x', None)]), inner_call_key=None), fun_body_sig='587e89b9790ab732b368f53e32a70e6f95d0bb73fb3b8ccf5058a7072d6fa55f', fun_return_sig='fc05cffdbcf25e9da8612976b2b072347eeed616492a7e8a7e9f8ed8604eb0a7', external_deps=[], parsed_body=[], store_path=None, fun_path=<dds_tests.test_default_args_basic._fun>)
INFO     dds._api:_api.py:140 Interaction tree:
INFO     dds._api:_api.py:141 `- Fun <dds_tests.test_default_args_basic._fun> /path <- fc05cffdbcf25e9da8612976b2b072347eeed616492a7e8a7e9f8ed8604eb0a7
INFO     dds._api:_api.py:141    `- Arg x: None
DEBUG    dds._api:_api.py:146 _eval_new_ctx:current_sig: fc05cffdbcf25e9da8612976b2b072347eeed616492a7e8a7e9f8ed8604eb0a7
DEBUG    dds._api:_api.py:148 _eval_new_ctx:Return cached signature fc05cffdbcf25e9da8612976b2b072347eeed616492a7e8a7e9f8ed8604eb0a7"""