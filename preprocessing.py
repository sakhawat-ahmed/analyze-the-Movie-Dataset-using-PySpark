from pyspark.sql.functions import col, year, to_date, expr

def load_and_clean_data(spark):
    movies_raw = spark.read.csv(
        "data/movies_metadata.csv",
        header=True,
        multiLine=True,
        escape='"',
        quote='"'
    )

    # Keep only numeric IDs
    movies_raw = movies_raw.filter(col("id").rlike("^[0-9]+$"))

    movies = movies_raw.select(
        expr("try_cast(id as int)").alias("movie_id"),
        expr("try_cast(budget as double)").alias("budget"),
        expr("try_cast(revenue as double)").alias("revenue"),
        expr("try_cast(vote_average as double)").alias("vote_average"),
        expr("try_cast(vote_count as int)").alias("vote_count"),
        expr("try_cast(runtime as double)").alias("runtime"),
        expr("try_cast(popularity as double)").alias("popularity"),
        to_date("release_date").alias("release_date")
    )

    # Drop rows where numeric casting failed
    movies = movies.dropna(subset=[
        "movie_id",
        "budget",
        "revenue",
        "vote_average",
        "popularity"
    ])

    movies = movies.withColumn(
        "release_year", year(col("release_date"))
    )

    ratings = spark.read.csv(
        "data/ratings.csv",
        header=True,
        inferSchema=True
    ).select(
        col("userId").cast("int"),
        col("movieId").cast("int"),
        col("rating").cast("float")
    )

    return movies, ratings
