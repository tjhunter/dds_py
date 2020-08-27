

@ks.describe(forget_args=["debug"])
def my_function(f: pys.Dataset, debug = False) -> int:
    return null

res = ks.cache(my_function, f=null)
res = ks.cache(my_function(f))
res = ks.keep("/test/value", my_function(f))
res = ks.keep("/test/value", my_function, f)


res = ks.load("/test/value")

model = ks.keep("/my_model", ks.aws.sagemaker(
    aws.s3_dataset("s3://bucket/path"),
    model=aws.sagemaker.pytorch({"num_epoch":3})
))


def preprocess(train_df, other_df) -> df:
    buckets = ks.keep(find_buckets(train_df))
    return ks.ml.bucketize(buckets, other_df)


def preprocess2(train_df):
    buckets = find_buckets(train_df)
    return lambda other_df: ks.ml.bucketize(buckets, other_df)


@ks.function()
def main():
    df = sp.load("...")
    (test_df, train_df) = ks.split(df)
    preprocessor = ks.cache(preprocess2(train_df))
    train_df = preprocessor(train_df)
    model = aws.sagemaker(train_df)

    test_df = preprocessor(test_df)
    ks.ml.apply(model, test_df)


def _loader():
    return sp.load("/...")

def load_data():
    return ks.keep("/...", _loader)


def transform(df):
    df["x"] = 3
    return df

def etl():
    df = load_data()
    df = transform(df)
    df2 = ks.load("/...")
    return df.count() + df2.count()


ks.eval(etl)


def main2():
    df = sp.load("...")
    df = df["xx"]




# *** model ***

const = "" + arg()


def f0():
    x = ks.load("")
    return x

def f1(arg1):
    x = ks.load("")
    x = ks.load(const)
    if arg1:
        x = ks.load(const)
    x = ks.load(arg1)  # forbidden? should be authorized
    x = ks.load(arg1 + "")  # forbidden
    x = ks.load(f3())  # forbidden?
    x = f0()
    return x


def f2(arg2):
    x = ks.keep(const, f1, arg1=arg2)
    x = ks.keep("", f1, arg1=arg2)
    return x

f2(None)
f2(None)

ks.keep(const, x)
f2(None)


