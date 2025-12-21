from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

def run_regression_models(movies):

    feature_cols = [
        "budget",
        "runtime",
        "popularity",
        "vote_count",
        "release_year"
    ]

    # 🔥 DROP rows with NULLs in features or label
    movies_clean = movies.dropna(
        subset=feature_cols + ["revenue", "vote_average"]
    )

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="error"   # now safe
    )

    data = assembler.transform(movies_clean)

    # ---------------- Revenue Prediction ----------------
    rev_data = data.select("features", "revenue")
    train, test = rev_data.randomSplit([0.8, 0.2], seed=42)

    lr_rev = LinearRegression(labelCol="revenue")
    rev_model = lr_rev.fit(train)

    predictions = rev_model.transform(test)

    evaluator = RegressionEvaluator(
        labelCol="revenue",
        predictionCol="prediction",
        metricName="rmse"
    )

    print("Revenue RMSE:", evaluator.evaluate(predictions))

    # ---------------- Vote Average Prediction ----------------
    vote_data = data.select("features", "vote_average")
    train, test = vote_data.randomSplit([0.8, 0.2], seed=42)

    lr_vote = LinearRegression(labelCol="vote_average")
    vote_model = lr_vote.fit(train)

    predictions = vote_model.transform(test)

    evaluator = RegressionEvaluator(
        labelCol="vote_average",
        predictionCol="prediction",
        metricName="rmse"
    )

    print("Vote Average RMSE:", evaluator.evaluate(predictions))
