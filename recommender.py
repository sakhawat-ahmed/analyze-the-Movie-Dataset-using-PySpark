from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col
from pyspark.ml.evaluation import RegressionEvaluator
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -----------------------------
# Memory-safe ALS training
# -----------------------------
def train_als_safe(ratings_df, max_sample=1000000, rank=10, reg_param=0.1, max_iter=10):
    """
    Train ALS safely on large datasets by sampling if necessary.
    
    Parameters:
        ratings_df: Spark DataFrame with columns ["userId", "movieId", "rating"]
        max_sample: maximum number of rows to use for training (int)
        rank: ALS rank
        reg_param: regularization parameter
        max_iter: max iterations
        
    Returns:
        ALSModel
    """
    row_count = ratings_df.count()
    logging.info(f"Total ratings rows: {row_count}")

    if row_count > max_sample:
        logging.warning(f"Sampling {max_sample} rows from {row_count} for memory safety")
        ratings_df = ratings_df.sample(withReplacement=False, fraction=max_sample / row_count, seed=42)
    
    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        rank=rank,
        regParam=reg_param,
        maxIter=max_iter,
        coldStartStrategy="drop",
        nonnegative=True
    )
    
    model = als.fit(ratings_df)
    logging.info("ALS model trained successfully.")
    return model

# -----------------------------
# ALS evaluation
# -----------------------------
def evaluate_als(model, ratings_df):
    predictions = model.transform(ratings_df)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    logging.info(f"ALS Model RMSE: {rmse:.4f}")
    return rmse

# -----------------------------
# Top-N similar movies
# -----------------------------
def get_top_n_similar_movies(movie_title, movies_df, als_model, top_n=10):
    from pyspark.sql import functions as F
    # Find movieId
    movie_row = movies_df.filter(col("title") == movie_title).select("movieId").limit(1).collect()
    if not movie_row:
        return None
    movie_id = movie_row[0]["movieId"]
    
    # Compute similarity via ALS factors
    movie_factors = als_model.itemFactors
    target_vector = movie_factors.filter(col("id") == movie_id).select("features").collect()
    if not target_vector:
        return None
    target_vector = target_vector[0]["features"]
    
    # Compute cosine similarity
    def cosine_similarity(features):
        dot = sum([f1 * f2 for f1, f2 in zip(features, target_vector)])
        norm1 = sum([f**2 for f in features]) ** 0.5
        norm2 = sum([f**2 for f in target_vector]) ** 0.5
        return float(dot / (norm1 * norm2)) if norm1 != 0 and norm2 != 0 else 0.0
    
    sim_udf = F.udf(cosine_similarity, "float")
    sims = movie_factors.withColumn("similarity", sim_udf(col("features")))
    
    top_similar_ids = sims.orderBy(col("similarity").desc()).filter(col("id") != movie_id).limit(top_n).select("id")
    return movies_df.join(top_similar_ids, movies_df.movieId == top_similar_ids.id, "inner").select("title", "movieId")

# -----------------------------
# Predict ratings for a user
# -----------------------------
def predict_user_ratings(user_id, unrated_movies_df, als_model):
    if unrated_movies_df.count() == 0:
        return unrated_movies_df
    user_df = unrated_movies_df.withColumn("userId", col("movieId") * 0 + user_id)
    predictions = als_model.transform(user_df).select("movieId", "prediction").orderBy(col("prediction").desc())
    return predictions
