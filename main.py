# movie_analysis_final_safe.py
import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, year, coalesce, lit, expr
from pyspark.sql.functions import try_to_date  # PySpark 3.5+
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from recommender import train_als_safe, evaluate_als, get_top_n_similar_movies, predict_user_ratings
import matplotlib.pyplot as plt

# -----------------------------
# Setup logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -----------------------------
# Initialize Spark with memory tuning
# -----------------------------
spark = SparkSession.builder \
    .appName("MovieLens Analysis Final Safe") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()
logging.info("Spark session initialized with memory tuning.")

# -----------------------------
# Load datasets
# -----------------------------
data_dir = "data"
movies = spark.read.csv(os.path.join(data_dir, "movies_metadata.csv"), header=True, inferSchema=True)
ratings = spark.read.csv(os.path.join(data_dir, "ratings.csv"), header=True, inferSchema=True)
logging.info("Datasets loaded.")

# -----------------------------
# Preprocess movies safely
# -----------------------------
movies = movies.withColumn("movieId", regexp_extract(col("id"), r"^\d+$", 0)) \
               .filter(col("movieId") != "") \
               .withColumn("movieId", col("movieId").cast("int"))

# Safe numeric casting
numeric_cols = ["budget", "revenue", "vote_average", "vote_count"]
for c in numeric_cols:
    movies = movies.withColumn(c, expr(f"try_cast({c} AS FLOAT)"))

# Safe release year extraction
movies = movies.withColumn(
    "release_year",
    coalesce(year(try_to_date(col("release_date"), "yyyy-MM-dd")), lit(0))
)

movies_reg = movies.select("budget", "vote_count", "release_year", "revenue", "vote_average") \
                   .na.drop() \
                   .cache()
logging.info(f"Movies preprocessed safely: {movies_reg.count()} rows ready for regression.")

# -----------------------------
# Preprocess ratings
# -----------------------------
ratings = ratings.select(
    col("userId").cast("int"),
    col("movieId").cast("int"),
    col("rating").cast("float")
).na.drop().cache()
logging.info(f"Ratings preprocessed: {ratings.count()} rows.")

# -----------------------------
# Task 1: Regression Models
# -----------------------------
def train_regression(features_cols, label_col):
    assembler = VectorAssembler(inputCols=features_cols, outputCol="features")
    data = assembler.transform(movies_reg).select("features", col(label_col).alias("label"))
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    lr = LinearRegression()
    model = lr.fit(train_data)
    predictions = model.transform(test_data)
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    return model, rmse

# Revenue prediction
model_rev, rmse_rev = train_regression(["budget", "vote_count", "release_year"], "revenue")
logging.info(f"Revenue Prediction RMSE: {rmse_rev:.2f}")

# Vote average prediction
model_vote, rmse_vote = train_regression(["budget", "vote_count", "release_year"], "vote_average")
logging.info(f"Vote Average Prediction RMSE: {rmse_vote:.2f}")

# -----------------------------
# Task 2: EDA / Data Visualization
# -----------------------------
movies_pd = movies_reg.select("vote_average", "budget", "vote_count", "release_year").toPandas()
movies_pd = movies_pd[movies_pd['release_year'] > 0]

# Scatter plot: Vote Average vs Budget
plt.figure(figsize=(8,5))
plt.scatter(movies_pd['budget'], movies_pd['vote_average'], alpha=0.3)
plt.xlabel("Budget")
plt.ylabel("Vote Average")
plt.title("Vote Average vs Movie Budget")
plt.savefig("vote_vs_budget.png")
plt.close()

# Scatter plot: Vote Average vs Vote Count
plt.figure(figsize=(8,5))
plt.scatter(movies_pd['vote_count'], movies_pd['vote_average'], alpha=0.3, color='orange')
plt.xlabel("Vote Count")
plt.ylabel("Vote Average")
plt.title("Vote Average vs Number of Votes")
plt.savefig("vote_vs_vote_count.png")
plt.close()

# Line plot: Average Vote by Release Year
vote_by_year = movies_pd.groupby("release_year")["vote_average"].mean()
plt.figure(figsize=(10,5))
vote_by_year.plot(kind="line", marker='o')
plt.xlabel("Release Year")
plt.ylabel("Average Vote")
plt.title("Average Vote by Release Year")
plt.savefig("vote_by_year.png")
plt.close()
logging.info("EDA plots generated and saved as PNGs.")

# -----------------------------
# Task 3: Collaborative Filtering (Memory-Safe ALS)
# -----------------------------
logging.info("Training memory-safe ALS model...")
als_model = train_als_safe(ratings)

evaluate_als(als_model, ratings)

# 3a: Top-N Similar Movies
movie_query = "The Godfather"
if movies.filter(col("title") == movie_query).count() == 0:
    logging.warning(f"Movie '{movie_query}' not found in dataset.")
else:
    similar_movies = get_top_n_similar_movies(movie_query, movies, als_model, top_n=10)
    if similar_movies and similar_movies.count() > 0:
        logging.info(f"Top-10 movies similar to '{movie_query}':")
        similar_movies.show(truncate=False)
    else:
        logging.warning(f"No similar movies found for '{movie_query}'.")

# 3b: Predict Ratings for a User
user_id = 1
all_movies = movies.select("movieId").distinct()
rated_movies = ratings.filter(col("userId") == user_id).select("movieId")
unrated_movies = all_movies.join(rated_movies, on="movieId", how="left_anti")

predictions = predict_user_ratings(user_id, unrated_movies, als_model)
if predictions.count() == 0:
    logging.warning(f"No predictions available for user {user_id}.")
else:
    logging.info(f"Predicted ratings for user {user_id}:")
    predictions.show(10, truncate=False)

# -----------------------------
# Stop Spark
# -----------------------------
spark.stop()
logging.info("Movie analysis completed safely! Plots saved and ALS predictions done.")
