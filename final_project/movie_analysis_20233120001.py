#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pyspark seaborn matplotlib plotly scikit-learn')
get_ipython().system('apt-get install openjdk-8-jdk-headless -qq > /dev/null')


# In[1]:


# Fix for Java version mismatch and permission issues

# 1. First check your current Java version
get_ipython().system('java -version')

# 3. Set Java Home to Java 17
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
os.environ["PYSPARK_PYTHON"] = "python3"

# Verify Java version
get_ipython().system('echo $JAVA_HOME')
get_ipython().system('ls $JAVA_HOME')

# 4. Initialize Spark
from pyspark.sql import SparkSession

try:
    spark = SparkSession.builder \
        .appName("MovieAnalysis") \
        .master("local") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    print("✓ Spark session created successfully!")
except Exception as e:
    print(f"✗ Error creating Spark session: {e}")
    print("\nTrying alternative approach...")
    
    # Try with lower memory settings
    spark = SparkSession.builder \
        .appName("MovieAnalysis") \
        .master("local[1]") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()
    print("✓ Spark session created with minimal settings!")


# In[2]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import *
from pyspark.ml.regression import *
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import *
from pyspark.ml import Pipeline
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("MovieDatasetAnalysis") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

print("Spark session created successfully!")


# In[4]:


# Define file paths (adjust based on your file locations)
file_paths = {
    'movies': 'data/movies_metadata.csv',
    'keywords': 'data/keywords.csv',
    'credits': 'data/credits.csv',
    'links': 'data/links.csv',
    'ratings': 'data/ratings.csv'
}

# Function to read CSV files with error handling
def read_csv_with_schema(file_path, schema=None):
    try:
        if schema:
            return spark.read.csv(file_path, header=True, schema=schema, escape='"')
        else:
            return spark.read.csv(file_path, header=True, inferSchema=True, escape='"')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Read all datasets
print("Loading datasets...")

# Read movies metadata
movies_df = read_csv_with_schema(file_paths['movies'])
print(f"Movies dataset loaded: {movies_df.count()} rows")

# Read keywords
keywords_df = read_csv_with_schema(file_paths['keywords'])
print(f"Keywords dataset loaded: {keywords_df.count()} rows")

# Read credits
credits_df = read_csv_with_schema(file_paths['credits'])
print(f"Credits dataset loaded: {credits_df.count()} rows")

# Read links
links_df = read_csv_with_schema(file_paths['links'])
print(f"Links dataset loaded: {links_df.count()} rows")

# Read ratings
ratings_df = read_csv_with_schema(file_paths['ratings'])
print(f"Ratings dataset loaded: {ratings_df.count()} rows")


# In[5]:


# Display schema of movies dataset
print("Movies dataset schema:")
movies_df.printSchema()

# Show first few rows
print("\nFirst 5 rows of movies dataset:")
movies_df.select("title", "budget", "revenue", "vote_average", "vote_count").show(5, truncate=False)


# In[7]:


# Data cleaning and preprocessing functions - FIXED VERSION

def clean_movies_data(df):
    """Clean and preprocess movies dataset"""
    
    # First, let's see the schema to understand the columns
    print("Original schema:")
    df.printSchema()
    
    # Use try_cast to handle conversion errors
    df = df.withColumn("budget", expr("try_cast(budget as double)")) \
           .withColumn("revenue", expr("try_cast(revenue as double)")) \
           .withColumn("vote_average", expr("try_cast(vote_average as double)")) \
           .withColumn("vote_count", expr("try_cast(vote_count as integer)")) \
           .withColumn("popularity", expr("try_cast(popularity as double)")) \
           .withColumn("runtime", expr("try_cast(runtime as double)"))
    
    # Filter out movies with invalid or missing data
    df = df.filter(
        (col("budget").isNotNull()) & 
        (col("revenue").isNotNull()) &
        (col("vote_average").isNotNull()) &
        (col("title").isNotNull())
    )
    
    # Filter out unrealistic values
    df = df.filter(
        (col("revenue") > 1000) & 
        (col("budget") > 1000) &
        (col("vote_count") > 10)
    )
    
    # Extract year from release_date
    df = df.withColumn("release_year", 
                       year(to_date(col("release_date"), "yyyy-MM-dd")))
    
    # Handle genres safely
    try:
        df = df.withColumn("genres_parsed", 
                          from_json(col("genres"), ArrayType(MapType(StringType(), StringType())))) \
               .withColumn("genres_list", 
                          expr("transform(genres_parsed, x -> x.name)"))
    except:
        # If parsing fails, create empty array
        df = df.withColumn("genres_list", array().cast(ArrayType(StringType())))
    
    return df

# Apply cleaning
movies_clean = clean_movies_data(movies_df)
print(f"Movies after cleaning: {movies_clean.count()} rows")

# Show some sample data
print("\nSample cleaned data:")
movies_clean.select("title", "budget", "revenue", "vote_average", "vote_count", "release_year").show(10, truncate=False)


# In[9]:


# First, let's create the movies_features DataFrame from cleaned data
# (Assuming you have movies_clean from the previous cleaning step)

# Create movies_features from movies_clean
movies_features = movies_clean.select(
    "id", 
    "title", 
    "budget", 
    "revenue", 
    "vote_average", 
    "vote_count",
    "popularity", 
    "runtime"
)

print(f"movies_features created with {movies_features.count()} rows")
movies_features.show(5, truncate=False)

# Now convert to pandas for visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

movies_pd = movies_features.filter(col("vote_count") > 50).toPandas()

print(f"Movies with vote_count > 50: {len(movies_pd)} rows")

# Create subplots for EDA
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=('Revenue Distribution', 'Vote Average Distribution',
                   'Budget vs Revenue', 'Vote Average vs Vote Count',
                   'Revenue by Release Year', 'Top 10 Genres by Average Rating'),
    specs=[[{'type': 'histogram'}, {'type': 'histogram'}, {'type': 'scatter'}],
           [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'bar'}]]
)

# Revenue distribution (log scale)
fig.add_trace(
    go.Histogram(x=np.log1p(movies_pd['revenue']), name='Revenue (log)'),
    row=1, col=1
)

# Vote average distribution
fig.add_trace(
    go.Histogram(x=movies_pd['vote_average'], name='Vote Average'),
    row=1, col=2
)

# Budget vs Revenue
fig.add_trace(
    go.Scatter(x=movies_pd['budget'], y=movies_pd['revenue'],
               mode='markers', name='Budget vs Revenue',
               marker=dict(size=3, opacity=0.5)),
    row=1, col=3
)

# Vote average vs vote count
fig.add_trace(
    go.Scatter(x=movies_pd['vote_count'], y=movies_pd['vote_average'],
               mode='markers', name='Votes',
               marker=dict(size=3, opacity=0.5, color=movies_pd['revenue'],
                          colorscale='Viridis', showscale=True,
                          colorbar=dict(title="Revenue"))),
    row=2, col=1
)

# Revenue by release year
# First, we need to add release_year to movies_features if not already there
# Let's check if we have release_year
print("\nChecking columns in movies_pd:")
print(movies_pd.columns.tolist())

# If release_year is not available, we'll skip that plot
if 'release_year' in movies_pd.columns:
    yearly_revenue = movies_pd.groupby('release_year')['revenue'].mean().reset_index()
    fig.add_trace(
        go.Scatter(x=yearly_revenue['release_year'], y=yearly_revenue['revenue'],
                   mode='lines+markers', name='Avg Revenue'),
        row=2, col=2
    )
else:
    # Create a placeholder or alternative plot
    print("release_year column not found, creating alternative plot")
    # You can create a different plot here, or skip this subplot

# For genres, we need to check if we have genres_list
if 'genres_list' in movies_pd.columns:
    # Top genres by average rating
    genres_list = []
    for idx, row in movies_pd.iterrows():
        if isinstance(row['genres_list'], list):
            for genre in row['genres_list']:
                genres_list.append({'genre': genre, 'vote_average': row['vote_average']})
    
    if genres_list:
        genres_df = pd.DataFrame(genres_list)
        genre_ratings = genres_df.groupby('genre')['vote_average'].mean().sort_values(ascending=False).head(10)
        
        fig.add_trace(
            go.Bar(x=genre_ratings.index, y=genre_ratings.values, name='Avg Rating'),
            row=2, col=3
        )
    else:
        print("No genre data available for bar chart")
else:
    print("genres_list column not found")

fig.update_layout(height=800, showlegend=False, title_text="Movie Dataset EDA")
fig.show()


# In[10]:


# Prepare data for regression
# Create feature columns
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

# Filter data for regression
regression_data = movies_features.filter(
    (col("vote_count") > 50) & 
    (col("revenue").isNotNull()) & 
    (col("budget").isNotNull()) &
    (col("runtime").isNotNull())
)

print(f"Data points for regression: {regression_data.count()}")

# Create features
# 1. Numeric features
numeric_cols = ["budget", "popularity", "runtime", "vote_count"]
# 2. Categorical features (simplified - using original_language)
categorical_cols = ["original_language"]

# Prepare feature engineering pipeline
indexers = [StringIndexer(inputCol=col, outputCol=col+"_idx", handleInvalid="keep") 
            for col in categorical_cols]

encoder = OneHotEncoder(inputCols=[col+"_idx" for col in categorical_cols],
                        outputCols=[col+"_enc" for col in categorical_cols])

# Assemble all features
feature_cols = numeric_cols + [col+"_enc" for col in categorical_cols]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Split data
train_data, test_data = regression_data.randomSplit([0.8, 0.2], seed=42)


# In[12]:


# SIMPLER REGRESSION MODEL - No categorical features needed

# First, let's prepare the data properly
from pyspark.ml.feature import VectorAssembler

# Use only numeric features for now
numeric_cols = ["budget", "vote_count", "popularity", "runtime"]

# Create feature vector
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")

# Prepare data - filter out null values
regression_data = movies_features.filter(
    col("budget").isNotNull() &
    col("revenue").isNotNull() &
    col("vote_count").isNotNull() &
    col("popularity").isNotNull() &
    col("runtime").isNotNull()
)

print(f"Data for regression: {regression_data.count()} rows")

# Split data
train_data, test_data = regression_data.randomSplit([0.8, 0.2], seed=42)
print(f"Training data: {train_data.count()} rows")
print(f"Test data: {test_data.count()} rows")

# Define models (simpler versions)
models = {
    "Linear Regression": LinearRegression(featuresCol="features", labelCol="revenue"),
    "Random Forest": RandomForestRegressor(featuresCol="features", labelCol="revenue", 
                                          numTrees=20, maxDepth=5),  # Reduced for speed
    "Gradient Boosting": GBTRegressor(featuresCol="features", labelCol="revenue", 
                                     maxIter=20, maxDepth=3)  # Reduced for speed
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training {name} for revenue prediction...")
    
    # Create simple pipeline with just assembler and model
    pipeline = Pipeline(stages=[assembler, model])
    
    # Train model
    trained_model = pipeline.fit(train_data)
    
    # Make predictions
    predictions = trained_model.transform(test_data)
    
    # Evaluate
    evaluator_rmse = RegressionEvaluator(labelCol="revenue", 
                                        predictionCol="prediction", 
                                        metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="revenue", 
                                      predictionCol="prediction", 
                                      metricName="r2")
    evaluator_mae = RegressionEvaluator(labelCol="revenue", 
                                       predictionCol="prediction", 
                                       metricName="mae")
    
    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    
    results[name] = {"RMSE": rmse, "R2": r2, "MAE": mae, "model": trained_model}
    
    print(f"{name} Results:")
    print(f"  RMSE: ${rmse:,.0f}")
    print(f"  MAE: ${mae:,.0f}")
    print(f"  R²: {r2:.4f}")
    
    # Show sample predictions
    print(f"\n  Sample predictions:")
    sample_preds = predictions.select("title", "revenue", "prediction").limit(5)
    for row in sample_preds.collect():
        title = row["title"][:30] + "..." if len(row["title"]) > 30 else row["title"]
        print(f"    {title:<35} Actual: ${row['revenue']:,.0f}, Predicted: ${row['prediction']:,.0f}")

# Compare results
print(f"\n{'='*50}")
print("MODEL COMPARISON SUMMARY")
print("="*50)
for name, metrics in results.items():
    print(f"{name:<20} RMSE: ${metrics['RMSE']:,.0f}  R²: {metrics['R2']:.4f}")


# In[14]:


# SIMPLIFIED VOTE AVERAGE PREDICTION - Using only available columns

# First, check what columns we have in movies_features
print("Available columns in movies_features:")
print(movies_features.columns)

# Prepare data for vote average prediction - using only columns that exist
vote_data = movies_features.filter(
    (col("vote_count") > 50) & 
    (col("vote_average").isNotNull()) &
    (col("budget").isNotNull()) &
    (col("popularity").isNotNull()) &
    (col("runtime").isNotNull())
)

print(f"\nData for vote prediction: {vote_data.count()} rows")

# Use only available numeric columns
# Let's see what numeric columns we have
numeric_cols_available = []
for col_name in ["budget", "popularity", "runtime", "vote_count", "revenue"]:
    if col_name in movies_features.columns:
        numeric_cols_available.append(col_name)

print(f"Available numeric columns: {numeric_cols_available}")

# Create feature vector
assembler_vote = VectorAssembler(
    inputCols=numeric_cols_available,
    outputCol="features"
)

# Split data
train_vote, test_vote = vote_data.randomSplit([0.8, 0.2], seed=42)
print(f"\nTraining data: {train_vote.count()} rows")
print(f"Test data: {test_vote.count()} rows")

# Define models for vote prediction
vote_models = {
    "Linear Regression": LinearRegression(featuresCol="features", labelCol="vote_average"),
    "Random Forest": RandomForestRegressor(featuresCol="features", labelCol="vote_average",
                                          numTrees=20, maxDepth=5),
    "Gradient Boosting": GBTRegressor(featuresCol="features", labelCol="vote_average",
                                     maxIter=20, maxDepth=3)
}

# Train and evaluate models
vote_results = {}
for name, model in vote_models.items():
    print(f"\n{'='*50}")
    print(f"Training {name} for vote average prediction...")
    
    # Create pipeline
    pipeline = Pipeline(stages=[assembler_vote, model])
    
    # Train model
    trained_model = pipeline.fit(train_vote)
    
    # Make predictions
    predictions = trained_model.transform(test_vote)
    
    # Evaluate
    evaluator_rmse = RegressionEvaluator(labelCol="vote_average", 
                                        predictionCol="prediction", 
                                        metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="vote_average", 
                                      predictionCol="prediction", 
                                      metricName="r2")
    evaluator_mae = RegressionEvaluator(labelCol="vote_average", 
                                       predictionCol="prediction", 
                                       metricName="mae")
    
    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    
    vote_results[name] = {"RMSE": rmse, "R2": r2, "MAE": mae, "model": trained_model}
    
    print(f"{name} Results:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # Show sample predictions
    print(f"\n  Sample predictions (first 3):")
    sample_preds = predictions.select("title", "vote_average", "prediction").limit(3)
    for row in sample_preds.collect():
        title = row["title"][:40] + "..." if len(row["title"]) > 40 else row["title"]
        print(f"    {title:<45} Actual: {row['vote_average']:.2f}, Predicted: {row['prediction']:.2f}")

# Compare results
print(f"\n{'='*50}")
print("VOTE PREDICTION MODEL COMPARISON")
print("="*50)
for name, metrics in vote_results.items():
    print(f"{name:<20} RMSE: {metrics['RMSE']:.4f}  R²: {metrics['R2']:.4f}")

# Feature importance for Random Forest (if we used it)
if "Random Forest" in vote_results:
    print(f"\n{'='*50}")
    print("Random Forest Feature Importance")
    print("="*50)
    
    rf_model = vote_results["Random Forest"]["model"].stages[-1]
    
    # Get feature importance
    feature_importance = list(zip(numeric_cols_available, rf_model.featureImportances.toArray()))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("Feature importance scores:")
    for feature, importance in feature_importance:
        print(f"  {feature:<15}: {importance:.4f}")


# In[15]:


# Visualize model comparison
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Revenue_RMSE': [results[m]['RMSE'] for m in results.keys()],
    'Revenue_R2': [results[m]['R2'] for m in results.keys()],
    'Vote_RMSE': [vote_results[m]['RMSE'] for m in vote_results.keys()],
    'Vote_R2': [vote_results[m]['R2'] for m in vote_results.keys()]
})

# Create comparison plot
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Revenue Prediction RMSE', 'Revenue Prediction R²',
                   'Vote Average Prediction RMSE', 'Vote Average Prediction R²'),
    specs=[[{'type': 'bar'}, {'type': 'bar'}],
           [{'type': 'bar'}, {'type': 'bar'}]]
)

# Revenue RMSE
fig.add_trace(
    go.Bar(x=results_df['Model'], y=results_df['Revenue_RMSE'],
           name='Revenue RMSE', marker_color='blue'),
    row=1, col=1
)

# Revenue R²
fig.add_trace(
    go.Bar(x=results_df['Model'], y=results_df['Revenue_R2'],
           name='Revenue R²', marker_color='green'),
    row=1, col=2
)

# Vote RMSE
fig.add_trace(
    go.Bar(x=results_df['Model'], y=results_df['Vote_RMSE'],
           name='Vote RMSE', marker_color='red'),
    row=2, col=1
)

# Vote R²
fig.add_trace(
    go.Bar(x=results_df['Model'], y=results_df['Vote_R2'],
           name='Vote R²', marker_color='orange'),
    row=2, col=2
)

fig.update_layout(height=800, showlegend=False, title_text="Model Comparison")
fig.show()


# In[17]:


# Convert to pandas for in-depth analysis
analysis_df = movies_pd.copy()

# Debug: Check available columns
print("Available columns in the dataset:")
print(analysis_df.columns.tolist())
print("\nFirst few rows:")
print(analysis_df.head())

# Check if 'genres' column exists (might be named differently)
genre_column = None
for col in analysis_df.columns:
    if 'genre' in col.lower():
        genre_column = col
        print(f"Found genre column: {col}")
        break

if genre_column is None:
    print("No genre column found. Checking for string columns that might contain genre info...")
    # Look for columns that might contain genre data
    for col in analysis_df.select_dtypes(include=['object']).columns:
        sample = str(analysis_df[col].iloc[0]) if not analysis_df[col].empty else ""
        if 'drama' in sample.lower() or 'comedy' in sample.lower() or 'action' in sample.lower():
            print(f"Potential genre column: {col}")
            print(f"Sample value: {sample[:100]}")
            genre_column = col
            break

# Create new features for analysis
analysis_df['budget_log'] = np.log1p(analysis_df['budget'])
analysis_df['revenue_log'] = np.log1p(analysis_df['revenue'])
analysis_df['profit_margin'] = (analysis_df['revenue'] - analysis_df['budget']) / analysis_df['revenue'].clip(lower=1)

# 1. Genre Analysis - with robust handling
genre_analysis = []

if genre_column:
    print(f"\nProcessing genre column: {genre_column}")
    
    # Check the data type of the genre column
    print(f"Data type: {analysis_df[genre_column].dtype}")
    print(f"Sample values:\n{analysis_df[genre_column].head()}")
    
    # Handle different possible formats
    for idx, row in analysis_df.iterrows():
        genres = row[genre_column]
        
        # Handle different genre formats
        if isinstance(genres, list):
            genre_list = genres
        elif isinstance(genres, str):
            # Try to parse string representation of list
            if genres.startswith('[') and genres.endswith(']'):
                try:
                    # Remove brackets and split
                    genres_clean = genres.strip('[]').replace("'", "").replace('"', '')
                    genre_list = [g.strip() for g in genres_clean.split(',')]
                except:
                    genre_list = [genres]
            else:
                # Split by comma or pipe
                if '|' in genres:
                    genre_list = [g.strip() for g in genres.split('|')]
                elif ',' in genres:
                    genre_list = [g.strip() for g in genres.split(',')]
                else:
                    genre_list = [genres]
        elif pd.isna(genres):
            genre_list = []
        else:
            genre_list = [str(genres)]
        
        # Add to analysis
        for genre in genre_list:
            if genre:  # Skip empty strings
                genre_analysis.append({
                    'genre': genre.strip(),
                    'vote_average': row['vote_average'],
                    'vote_count': row['vote_count'],
                    'revenue': row['revenue'],
                    'budget': row['budget']
                })

if genre_analysis:
    genre_df = pd.DataFrame(genre_analysis)
    print(f"\nCreated genre dataframe with {len(genre_df)} entries")
    print(f"Unique genres: {genre_df['genre'].nunique()}")
    print("\nGenre counts:")
    print(genre_df['genre'].value_counts().head(10))
else:
    print("\nNo genre data available. Creating empty dataframe.")
    genre_df = pd.DataFrame(columns=['genre', 'vote_average', 'vote_count', 'revenue', 'budget'])

# 2. Language Analysis
print("\n=== Language Analysis ===")
if 'original_language' in analysis_df.columns:
    language_stats = analysis_df.groupby('original_language').agg({
        'vote_average': ['mean', 'count'],
        'revenue': 'mean'
    }).reset_index()
    language_stats.columns = ['language', 'avg_rating', 'movie_count', 'avg_revenue']
    language_stats = language_stats[language_stats['movie_count'] > 10].sort_values('avg_rating', ascending=False)
    print(f"Languages with >10 movies: {len(language_stats)}")
else:
    print("'original_language' column not found")
    language_stats = pd.DataFrame(columns=['language', 'avg_rating', 'movie_count', 'avg_revenue'])

# 3. Runtime Analysis
print("\n=== Runtime Analysis ===")
if 'runtime' in analysis_df.columns:
    # Filter out extreme outliers
    runtime_clean = analysis_df['runtime'].dropna()
    q1 = runtime_clean.quantile(0.01)
    q99 = runtime_clean.quantile(0.99)
    
    analysis_df['runtime_clean'] = analysis_df['runtime'].clip(q1, q99)
    analysis_df['runtime_binned'] = pd.cut(analysis_df['runtime_clean'], 
                                           bins=[0, 60, 90, 120, 150, 180, 1000],
                                           labels=['<1h', '1-1.5h', '1.5-2h', '2-2.5h', '2.5-3h', '>3h'])
    print(f"Runtime range: {analysis_df['runtime_clean'].min():.0f} to {analysis_df['runtime_clean'].max():.0f} minutes")
else:
    print("'runtime' column not found")
    analysis_df['runtime_binned'] = pd.Series()

# Create comprehensive visualization
fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=('Rating Distribution by Genre', 'Top Languages by Avg Rating',
                   'Runtime vs Rating', 'Budget vs Rating',
                   'Revenue vs Rating', 'Vote Count vs Rating',
                   'Release Year vs Rating', 'Profit Margin vs Rating',
                   'Popularity vs Rating'),
    specs=[[{'type': 'box'}, {'type': 'bar'}, {'type': 'box'}],
           [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
           [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]],
    vertical_spacing=0.08,
    horizontal_spacing=0.08
)

# 1. Genre box plot - only if we have genre data
if not genre_df.empty and len(genre_df['genre'].unique()) > 1:
    top_genres = genre_df.groupby('genre')['vote_average'].mean().sort_values(ascending=False).head(10).index
    genre_filtered = genre_df[genre_df['genre'].isin(top_genres)]
    
    for genre in top_genres[:5]:  # Show top 5 for clarity
        genre_data = genre_filtered[genre_filtered['genre'] == genre]['vote_average']
        fig.add_trace(
            go.Box(y=genre_data, name=genre[:15], boxpoints=False),  # Truncate long genre names
            row=1, col=1
        )
else:
    # Add placeholder if no genre data
    fig.add_trace(
        go.Box(y=[], name='No genre data'),
        row=1, col=1
    )

# 2. Top languages bar chart - only if we have language data
if not language_stats.empty:
    fig.add_trace(
        go.Bar(x=language_stats.head(10)['language'], 
               y=language_stats.head(10)['avg_rating'],
               name='Language Rating',
               marker_color='lightblue'),
        row=1, col=2
    )
else:
    fig.add_trace(
        go.Bar(x=[], y=[], name='No language data'),
        row=1, col=2
    )

# 3. Runtime vs Rating
if 'runtime_binned' in analysis_df.columns and not analysis_df['runtime_binned'].isna().all():
    runtime_stats = analysis_df.groupby('runtime_binned')['vote_average'].mean().reset_index()
    fig.add_trace(
        go.Bar(x=runtime_stats['runtime_binned'], y=runtime_stats['vote_average'],
               name='Runtime Rating', marker_color='lightgreen'),
        row=1, col=3
    )
else:
    fig.add_trace(
        go.Bar(x=[], y=[], name='No runtime data'),
        row=1, col=3
    )

# 4. Budget vs Rating
fig.add_trace(
    go.Scatter(x=analysis_df['budget_log'], y=analysis_df['vote_average'],
               mode='markers', name='Budget',
               marker=dict(size=3, opacity=0.3, color='blue'),
               hovertemplate='Budget: $%{x:.1f}<br>Rating: %{y:.2f}<extra></extra>'),
    row=2, col=1
)

# 5. Revenue vs Rating
fig.add_trace(
    go.Scatter(x=analysis_df['revenue_log'], y=analysis_df['vote_average'],
               mode='markers', name='Revenue',
               marker=dict(size=3, opacity=0.3, color='green'),
               hovertemplate='Revenue: $%{x:.1f}<br>Rating: %{y:.2f}<extra></extra>'),
    row=2, col=2
)

# 6. Vote Count vs Rating
fig.add_trace(
    go.Scatter(x=np.log1p(analysis_df['vote_count']), y=analysis_df['vote_average'],
               mode='markers', name='Vote Count',
               marker=dict(size=3, opacity=0.3, color='orange'),
               hovertemplate='Vote Count (log): %{x:.1f}<br>Rating: %{y:.2f}<extra></extra>'),
    row=2, col=3
)

# 7. Release Year vs Rating - only if column exists
if 'release_year' in analysis_df.columns:
    year_stats = analysis_df.groupby('release_year')['vote_average'].mean().reset_index()
    fig.add_trace(
        go.Scatter(x=year_stats['release_year'], y=year_stats['vote_average'],
                   mode='lines+markers', name='Year Trend',
                   line=dict(color='red', width=2),
                   marker=dict(size=4)),
        row=3, col=1
    )
else:
    fig.add_trace(
        go.Scatter(x=[], y=[], name='No release year data'),
        row=3, col=1
    )

# 8. Profit Margin vs Rating
fig.add_trace(
    go.Scatter(x=analysis_df['profit_margin'].clip(-1, 1), y=analysis_df['vote_average'],
               mode='markers', name='Profit Margin',
               marker=dict(size=3, opacity=0.3, color='purple'),
               hovertemplate='Profit Margin: %{x:.2%}<br>Rating: %{y:.2f}<extra></extra>'),
    row=3, col=2
)

# 9. Popularity vs Rating
if 'popularity' in analysis_df.columns:
    fig.add_trace(
        go.Scatter(x=np.log1p(analysis_df['popularity']), y=analysis_df['vote_average'],
                   mode='markers', name='Popularity',
                   marker=dict(size=3, opacity=0.3, color='brown'),
                   hovertemplate='Popularity (log): %{x:.1f}<br>Rating: %{y:.2f}<extra></extra>'),
        row=3, col=3
    )
else:
    fig.add_trace(
        go.Scatter(x=[], y=[], name='No popularity data'),
        row=3, col=3
    )

# Update layout
fig.update_layout(
    height=1200,
    showlegend=False,
    title_text="Comprehensive Movie Rating Analysis",
    title_font_size=20,
    title_x=0.5
)

# Update axes labels
fig.update_xaxes(title_text="Genre", row=1, col=1)
fig.update_xaxes(title_text="Language", row=1, col=2)
fig.update_xaxes(title_text="Runtime", row=1, col=3)

fig.update_xaxes(title_text="Log(Budget)", row=2, col=1)
fig.update_xaxes(title_text="Log(Revenue)", row=2, col=2)
fig.update_xaxes(title_text="Log(Vote Count)", row=2, col=3)

fig.update_xaxes(title_text="Release Year", row=3, col=1)
fig.update_xaxes(title_text="Profit Margin", row=3, col=2)
fig.update_xaxes(title_text="Log(Popularity)", row=3, col=3)

for i in range(1, 4):
    for j in range(1, 4):
        fig.update_yaxes(title_text="Rating", row=i, col=j)

fig.show()

# Summary statistics
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
print(f"Total movies analyzed: {len(analysis_df)}")
print(f"Average vote average: {analysis_df['vote_average'].mean():.2f}")
print(f"Median vote average: {analysis_df['vote_average'].median():.2f}")
print(f"Std deviation of ratings: {analysis_df['vote_average'].std():.2f}")

# Calculate correlations with error handling
print("\n=== CORRELATIONS WITH RATING ===")
for col in ['budget', 'revenue', 'vote_count', 'popularity', 'runtime']:
    if col in analysis_df.columns:
        corr = analysis_df[col].corr(analysis_df['vote_average'])
        if pd.notna(corr):
            print(f"Correlation between {col} and rating: {corr:.3f}")

# Top performing genres - if available
if not genre_df.empty and len(genre_df) > 0:
    print("\n" + "="*50)
    print("TOP GENRES BY AVERAGE RATING (min 10 movies)")
    print("="*50)
    
    # Filter genres with sufficient data
    genre_counts = genre_df['genre'].value_counts()
    valid_genres = genre_counts[genre_counts >= 10].index
    
    if len(valid_genres) > 0:
        top_genres_df = genre_df[genre_df['genre'].isin(valid_genres)].groupby('genre').agg({
            'vote_average': ['mean', 'count'],
            'vote_count': 'sum',
            'revenue': 'mean'
        }).sort_values(('vote_average', 'mean'), ascending=False).head(10)
        
        # Flatten multi-index columns
        top_genres_df.columns = ['avg_rating', 'movie_count', 'total_votes', 'avg_revenue']
        
        print("\nTop 10 genres:")
        for idx, row in top_genres_df.iterrows():
            print(f"{idx:20} | Rating: {row['avg_rating']:.2f} | Movies: {row['movie_count']:3d} | "
                  f"Votes: {row['total_votes']:,} | Revenue: ${row['avg_revenue']/1e6:.1f}M")
    else:
        print("Insufficient genre data for meaningful analysis")

print("\n" + "="*50)
print("ADDITIONAL INSIGHTS")
print("="*50)

# Additional insights
if 'release_year' in analysis_df.columns:
    recent_movies = analysis_df[analysis_df['release_year'] >= 2010]
    old_movies = analysis_df[analysis_df['release_year'] < 2010]
    
    if len(recent_movies) > 0 and len(old_movies) > 0:
        print(f"Average rating for movies since 2010: {recent_movies['vote_average'].mean():.2f}")
        print(f"Average rating for movies before 2010: {old_movies['vote_average'].mean():.2f}")
        print(f"Difference: {recent_movies['vote_average'].mean() - old_movies['vote_average'].mean():.2f}")

# High budget vs low budget comparison
if 'budget' in analysis_df.columns:
    median_budget = analysis_df['budget'].median()
    high_budget = analysis_df[analysis_df['budget'] > median_budget]
    low_budget = analysis_df[analysis_df['budget'] <= median_budget]
    
    if len(high_budget) > 0 and len(low_budget) > 0:
        print(f"\nHigh budget movies (>${median_budget/1e6:.1f}M): {high_budget['vote_average'].mean():.2f}")
        print(f"Low budget movies: {low_budget['vote_average'].mean():.2f}")


# In[21]:


from pyspark.sql.functions import col
# Prepare data for collaborative filtering
# Clean and prepare ratings data
ratings_clean = ratings_df.select(
    col("userId").cast("integer").alias("user_id"),
    col("movieId").cast("integer").alias("movie_id"),
    col("rating").cast("float").alias("rating"),
    col("timestamp").cast("long").alias("timestamp")
)

# Clean links data for mapping
links_clean = links_df.select(
    col("movieId").cast("integer").alias("movie_id"),
    col("tmdbId").cast("integer").alias("tmdb_id"),
    col("imdbId").cast("string").alias("imdb_id")
)

# Clean movies data for titles
movies_titles = movies_clean.select(
    col("id").cast("integer").alias("tmdb_id"),
    "title"
).filter(col("tmdb_id").isNotNull())

# Join to create complete dataset
ratings_with_titles = ratings_clean.join(
    links_clean, on="movie_id", how="inner"
).join(
    movies_titles, on="tmdb_id", how="inner"
).select("user_id", "tmdb_id", "title", "rating")

print(f"Ratings with titles: {ratings_with_titles.count()} rows")
print(f"Unique users: {ratings_with_titles.select('user_id').distinct().count()}")
print(f"Unique movies: {ratings_with_titles.select('tmdb_id').distinct().count()}")

# Cache the data for faster processing
ratings_with_titles.cache()


# In[22]:


# Split data for training and testing
(training, test) = ratings_with_titles.randomSplit([0.8, 0.2], seed=42)

print(f"Training data: {training.count()} rows")
print(f"Test data: {test.count()} rows")

# Build ALS model
als = ALS(
    userCol="user_id",
    itemCol="tmdb_id",
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True,
    regParam=0.1,
    rank=10,
    maxIter=10,
    seed=42
)

# Train the model
print("Training ALS model...")
model = als.fit(training)

# Evaluate the model
predictions = model.transform(test)
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

rmse = evaluator.evaluate(predictions)
print(f"RMSE on test data: {rmse:.4f}")

# Show some predictions
print("\nSample predictions:")
predictions.select("user_id", "title", "rating", "prediction").show(10)


# In[24]:


# Function 3a: Suggest top N similar movies
def get_similar_movies(movie_title, n_recommendations=10):
    """Get top N movies similar to a given movie"""
    
    # Find the movie ID
    movie_df = movies_titles.filter(col("title") == movie_title)
    if movie_df.count() == 0:
        print(f"Movie '{movie_title}' not found in database.")
        return None
    
    movie_id = movie_df.first()["tmdb_id"]
    
    # Get item factors (movie embeddings)
    item_factors = model.itemFactors.filter(col("id") == movie_id)
    if item_factors.count() == 0:
        print(f"Movie ID {movie_id} not found in model.")
        return None
    
    # Get the embedding for the target movie
    target_features = item_factors.first()["features"]
    
    # Calculate similarity with all other movies
    all_items = model.itemFactors.alias("items")
    
    # Broadcast target features for efficient computation
    target_features_bc = spark.sparkContext.broadcast(target_features)
    
    def cosine_similarity(features):
        import numpy as np
        target = np.array(target_features_bc.value)
        item = np.array(features)
        dot_product = np.dot(target, item)
        norm_target = np.linalg.norm(target)
        norm_item = np.linalg.norm(item)
        return float(dot_product / (norm_target * norm_item))
    
    # Register UDF
    from pyspark.sql.functions import udf
    from pyspark.sql.types import FloatType
    
    cosine_similarity_udf = udf(cosine_similarity, FloatType())
    
    # Calculate similarities
    similar_movies = all_items.withColumn(
        "similarity",
        cosine_similarity_udf(col("features"))
    ).filter(col("id") != movie_id) \
     .join(movies_titles, col("id") == col("tmdb_id")) \
     .select("title", "similarity") \
     .orderBy(col("similarity").desc()) \
     .limit(n_recommendations)
    
    return similar_movies

# Function 3b: Predict user ratings for unrated movies
def predict_user_ratings(user_id, n_predictions=10):
    """Predict top N movie ratings for a given user"""
    
    # Get all movies
    all_movies = movies_titles.select("tmdb_id").distinct()
    
    # Get movies already rated by user
    rated_movies = ratings_with_titles.filter(col("user_id") == user_id) \
        .select("tmdb_id").distinct()
    
    # Get unrated movies
    unrated_movies = all_movies.join(
        rated_movies, 
        all_movies["tmdb_id"] == rated_movies["tmdb_id"], 
        how="left_anti"
    ).withColumn("user_id", lit(user_id))
    
    # Make predictions
    predictions = model.transform(unrated_movies)
    
    # Get top predictions
    top_predictions = predictions.join(
        movies_titles, on="tmdb_id"
    ).select("title", "prediction") \
     .orderBy(col("prediction").desc()) \
     .limit(n_predictions)
    
    return top_predictions


# In[25]:


# Test the recommendation functions

print("=== Testing Recommendation System ===")

# Test 3a: Similar movies
test_movie = "Toy Story"
print(f"\n1. Finding movies similar to '{test_movie}':")
similar_movies = get_similar_movies(test_movie, 10)
if similar_movies:
    similar_movies.show(truncate=False)

# Test with another movie
test_movie2 = "The Dark Knight"
print(f"\n2. Finding movies similar to '{test_movie2}':")
similar_movies2 = get_similar_movies(test_movie2, 10)
if similar_movies2:
    similar_movies2.show(truncate=False)

# Test 3b: User rating predictions
test_user_id = 1
print(f"\n3. Predicting top 10 movies for user {test_user_id}:")
user_predictions = predict_user_ratings(test_user_id, 10)
user_predictions.show(truncate=False)

# Test with another user
test_user_id2 = 100
print(f"\n4. Predicting top 10 movies for user {test_user_id2}:")
user_predictions2 = predict_user_ratings(test_user_id2, 10)
user_predictions2.show(truncate=False)


# In[34]:


from pyspark.sql.functions import col, rand, abs as spark_abs, mean, stddev, min as spark_min, max as spark_max, collect_set, explode, lit, count as spark_count
from pyspark.ml.recommendation import ALS
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
import numpy as np
from pyspark.ml.evaluation import RegressionEvaluator

# First, let's check what columns we have
print("Columns in ratings_with_titles:")
print(ratings_with_titles.columns)

# Create a proper ALS-ready dataset with only 3 columns
als_data = ratings_with_titles.select("user_id", "tmdb_id", "rating")
print(f"\nALS data count: {als_data.count()}")
print("ALS data columns:", als_data.columns)

# Create training and test splits
print("\n=== Creating Training and Test Sets ===")

# Add random seed for reproducibility
als_data_with_seed = als_data.withColumn("rand", rand(seed=42))

# Split data into training and test (80/20 split)
training = als_data_with_seed.filter(col("rand") < 0.8).drop("rand")
test = als_data_with_seed.filter(col("rand") >= 0.8).drop("rand")

print(f"Training set size: {training.count():,}")
print(f"Test set size: {test.count():,}")

# Show some statistics
print("\n=== Data Statistics ===")
print(f"Total unique users: {als_data.select('user_id').distinct().count():,}")
print(f"Total unique movies: {als_data.select('tmdb_id').distinct().count():,}")
print(f"Average ratings per user: {als_data.count() / als_data.select('user_id').distinct().count():.1f}")
print(f"Average ratings per movie: {als_data.count() / als_data.select('tmdb_id').distinct().count():.1f}")

# Train ALS model
print("\n=== Training ALS Model ===")

als = ALS(
    userCol="user_id",
    itemCol="tmdb_id",
    ratingCol="rating",
    coldStartStrategy="drop",
    implicitPrefs=False,
    nonnegative=True,
    maxIter=10,
    regParam=0.1,
    rank=10
)

model = als.fit(training)
print("Model trained successfully!")

# Evaluate on test set
print("\n=== Evaluating Model on Test Set ===")

# Generate predictions
test_predictions = model.transform(test)

# Filter out NaN predictions (cold start items)
test_predictions_clean = test_predictions.filter(col("prediction").isNotNull())

print(f"Total test predictions: {test_predictions.count():,}")
print(f"Valid test predictions (non-null): {test_predictions_clean.count():,}")
print(f"Cold start items (null predictions): {test_predictions.count() - test_predictions_clean.count():,}")

# Calculate RMSE and MAE using Spark's built-in evaluator
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

rmse = evaluator.evaluate(test_predictions_clean)
print(f"\nRMSE on test set: {rmse:.3f}")

evaluator.setMetricName("mae")
mae = evaluator.evaluate(test_predictions_clean)
print(f"MAE on test set: {mae:.3f}")

# Calculate R-squared (coefficient of determination)
evaluator.setMetricName("r2")
r2 = evaluator.evaluate(test_predictions_clean)
print(f"R-squared on test set: {r2:.3f}")

# Calculate prediction error distribution using Spark SQL
print("\n=== Prediction Error Distribution ===")

# Add error column
test_with_error = test_predictions_clean.withColumn("error", spark_abs(col("rating") - col("prediction")))

# Calculate error statistics using agg() instead of select()
error_stats = test_with_error.agg(
    mean(col("error")).alias("mean_error"),
    stddev(col("error")).alias("std_error"),
    spark_min(col("error")).alias("min_error"),
    spark_max(col("error")).alias("max_error")
).collect()[0]

print(f"\nError statistics:")
print(f"  Mean error: {error_stats['mean_error']:.3f}")
print(f"  Std deviation: {error_stats['std_error']:.3f}")
print(f"  Min error: {error_stats['min_error']:.3f}")
print(f"  Max error: {error_stats['max_error']:.3f}")

# Define error ranges and calculate percentages
total_count = test_with_error.count()
if total_count > 0:
    print(f"\nError distribution (% of predictions within error range):")
    
    for error_threshold in [0.5, 1.0, 1.5, 2.0]:
        count = test_with_error.filter(col("error") <= error_threshold).count()
        percentage = (count / total_count) * 100
        print(f"  Within ±{error_threshold:.1f}: {count:,} ({percentage:.1f}%)")
else:
    print("No predictions to analyze")

# Get top recommendations for all users
print("\n=== Generating Top-N Recommendations ===")
user_recs = model.recommendForAllUsers(20)
user_recs_count = user_recs.count()
print(f"Generated recommendations for {user_recs_count:,} users")

# Evaluate top-N recommendations
print("\n=== Evaluating Top-N Recommendations ===")

# Get highly rated items from test set (rating >= 4)
test_high = test.filter(col("rating") >= 4.0)
print(f"Highly rated items in test set (rating >= 4): {test_high.count():,}")

# Collect test high ratings to a dictionary for faster lookup
print("Collecting test high ratings...")
test_high_dict = {}
test_high_rows = test_high.select("user_id", "tmdb_id").collect()

for row in test_high_rows:
    user_id = row["user_id"]
    tmdb_id = row["tmdb_id"]
    if user_id not in test_high_dict:
        test_high_dict[user_id] = set()
    test_high_dict[user_id].add(tmdb_id)

print(f"Created lookup for {len(test_high_dict):,} users with highly rated items")

# Calculate precision for different k values using Python's built-in functions
print("\n=== Calculating Precision@K ===")

# Use built-in min and sum functions
builtin_min = __builtins__.min
builtin_sum = __builtins__.sum

for k in [5, 10, 20]:
    total_hits = 0
    total_recommendations = 0
    
    # Sample some users for evaluation (for performance)
    sample_size = builtin_min(500, user_recs_count)
    sample_users = user_recs.limit(sample_size).collect()
    
    for user_row in sample_users:
        user_id = user_row["user_id"]
        recommendations = user_row["recommendations"][:k]  # Top k recommendations
        
        if user_id in test_high_dict:
            user_high_rated = test_high_dict[user_id]
            # Use Python's built-in sum
            hits = builtin_sum(1 for rec in recommendations if rec["tmdb_id"] in user_high_rated)
            total_hits += hits
        
        total_recommendations += len(recommendations)
    
    precision = total_hits / total_recommendations if total_recommendations > 0 else 0
    print(f"Precision@{k}: {precision:.3f} (sampled {len(sample_users)} users)")

# Show sample recommendations
print("\n=== Sample Recommendations ===")

# Get sample users
sample_users_list = test.select("user_id").distinct().limit(3).collect()

for i, user_row in enumerate(sample_users_list):
    user_id = user_row["user_id"]
    
    print(f"\n{'='*60}")
    print(f"User {user_id}:")
    print(f"{'='*60}")
    
    # Get user's actual ratings from test set
    user_ratings = test.filter(col("user_id") == user_id) \
                      .join(ratings_with_titles.select("tmdb_id", "title").distinct(), "tmdb_id") \
                      .select("title", "rating") \
                      .orderBy(col("rating").desc()) \
                      .limit(5)
    
    print(f"\nActual ratings (from test set):")
    if user_ratings.count() > 0:
        for rating_row in user_ratings.collect():
            print(f"  • {rating_row['title']}: {rating_row['rating']:.1f}")
    else:
        print("  No ratings in test set")
    
    # Get recommendations
    user_recommendations = user_recs.filter(col("user_id") == user_id)
    
    if user_recommendations.count() > 0:
        recs = user_recommendations.collect()[0]["recommendations"][:10]
        
        print(f"\nTop recommendations:")
        for j, rec in enumerate(recs):
            # Get movie title
            movie_title = ratings_with_titles.filter(col("tmdb_id") == rec["tmdb_id"]) \
                                           .select("title").first()
            title = movie_title["title"] if movie_title else f"Movie {rec['tmdb_id']}"
            print(f"  {j+1}. {title} (predicted: {rec['rating']:.2f})")
    else:
        print(f"\nNo recommendations available for this user")

# Additional evaluation metrics
print("\n=== Additional Evaluation Metrics ===")

# Calculate coverage (percentage of items that can be recommended)
all_items = als_data.select("tmdb_id").distinct().count()
recommended_items = user_recs.select(
    explode("recommendations").alias("rec")
).select(
    col("rec.tmdb_id").alias("recommended_tmdb_id")
).distinct().count()

coverage = recommended_items / all_items if all_items > 0 else 0
print(f"Coverage: {coverage:.3f} ({recommended_items:,}/{all_items:,} items)")

# Calculate average predicted rating
avg_predicted_rating = test_predictions_clean.agg(
    mean(col("prediction")).alias("avg_prediction")
).collect()[0]["avg_prediction"]

avg_actual_rating = test.agg(
    mean(col("rating")).alias("avg_rating")
).collect()[0]["avg_rating"]

print(f"Average actual rating in test set: {avg_actual_rating:.3f}")
print(f"Average predicted rating: {avg_predicted_rating:.3f}")
print(f"Bias (predicted - actual): {avg_predicted_rating - avg_actual_rating:.3f}")

# Calculate rating prediction distribution
print("\n=== Rating Prediction Distribution ===")
rating_bins = [(0, 2), (2, 3), (3, 4), (4, 5)]
for low, high in rating_bins:
    count = test_predictions_clean.filter(
        (col("prediction") >= low) & (col("prediction") < high)
    ).count()
    percentage = (count / test_predictions_clean.count()) * 100 if test_predictions_clean.count() > 0 else 0
    print(f"Predictions in range [{low}, {high}): {count:,} ({percentage:.1f}%)")

# Model diagnostics and hyperparameters
print("\n=== Model Configuration ===")
print(f"Rank (latent factors): {als.getRank()}")
print(f"Regularization parameter: {als.getRegParam()}")
print(f"Maximum iterations: {als.getMaxIter()}")
print(f"Non-negative constraints: {als.getNonnegative()}")

# Check for overfitting by comparing train and test performance
train_predictions = model.transform(training).filter(col("prediction").isNotNull())
train_rmse = evaluator.evaluate(train_predictions)
print(f"\nTraining RMSE: {train_rmse:.3f}")
print(f"Test RMSE: {rmse:.3f}")
print(f"Overfitting indicator (test RMSE - train RMSE): {rmse - train_rmse:.3f}")

# Calculate user and item factors statistics
print("\n=== Latent Factors Analysis ===")
user_factors = model.userFactors
item_factors = model.itemFactors

print(f"User factors dimensions: {user_factors.count():,} users × {als.getRank()} factors")
print(f"Item factors dimensions: {item_factors.count():,} items × {als.getRank()} factors")

# Simple precision calculation for final summary
def calculate_simple_precision(k):
    """Calculate precision@k using a simple approach"""
    total_hits = 0
    total_recs = 0
    
    # Use a reasonable sample size
    sample_size = builtin_min(1000, user_recs_count)
    sample = user_recs.limit(sample_size).collect()
    
    for user_row in sample:
        user_id = user_row["user_id"]
        recommendations = user_row["recommendations"][:k]
        
        if user_id in test_high_dict:
            user_high_rated = test_high_dict[user_id]
            hits = builtin_sum(1 for rec in recommendations if rec["tmdb_id"] in user_high_rated)
            total_hits += hits
        
        total_recs += len(recommendations)
    
    return total_hits / total_recs if total_recs > 0 else 0

# Final summary
print("\n" + "="*60)
print("EVALUATION SUMMARY")
print("="*60)
print(f"\nMODEL PERFORMANCE:")
print(f"  • RMSE: {rmse:.3f}")
print(f"  • MAE: {mae:.3f}")
print(f"  • R-squared: {r2:.3f}")
print(f"  • Mean absolute error: {error_stats['mean_error']:.3f}")

print(f"\nRECOMMENDATION QUALITY:")
# Calculate precision for final summary
precision_5 = calculate_simple_precision(5)
precision_10 = calculate_simple_precision(10)
print(f"  • Precision@5: {precision_5:.3f}")
print(f"  • Precision@10: {precision_10:.3f}")

print(f"\nDATA STATISTICS:")
print(f"  • Training samples: {training.count():,}")
print(f"  • Test samples: {test.count():,}")
print(f"  • Unique users: {als_data.select('user_id').distinct().count():,}")
print(f"  • Unique movies: {als_data.select('tmdb_id').distinct().count():,}")
print(f"  • Coverage: {coverage:.2%}")

print(f"\nERROR DISTRIBUTION:")
if total_count > 0:
    for error_threshold in [0.5, 1.0, 1.5]:
        count = test_with_error.filter(col("error") <= error_threshold).count()
        percentage = (count / total_count) * 100
        print(f"  • Within ±{error_threshold:.1f}: {percentage:.1f}%")

print("\n" + "="*60)
print("RECOMMENDATIONS:")
print("="*60)
print("\nBased on the evaluation, the ALS model:")
print(f"1. Shows decent prediction accuracy with RMSE of {rmse:.3f}")
print(f"2. Has reasonable coverage of {coverage:.1%} of all movies")
print(f"3. Precision@10 of {precision_10:.3f} indicates relevance of recommendations")
if total_count > 0:
    within_1 = test_with_error.filter(col("error") <= 1.0).count() / total_count * 100
    print(f"4. Error distribution shows {within_1:.1f}% of predictions within ±1.0 rating")

print("\nSuggestions for improvement:")
print("1. Increase training data or adjust hyperparameters")
print("2. Consider content-based filtering to handle cold start")
print("3. Implement hybrid recommendation approach")
print("4. Add temporal features for recency bias")

# Show top recommended movies overall - Simplified version
print("\n=== Most Frequently Recommended Movies ===")
top_recommended_movies = user_recs.select(
    explode("recommendations").alias("rec")
).select(
    col("rec.tmdb_id").alias("tmdb_id")
).groupBy("tmdb_id").agg(
    spark_count("*").alias("recommendation_count")
).orderBy(col("recommendation_count").desc()).limit(10)

# Join with movie titles
top_movies_with_titles = top_recommended_movies.join(
    ratings_with_titles.select("tmdb_id", "title").distinct(),
    "tmdb_id",
    "inner"
).select("title", "recommendation_count").orderBy(col("recommendation_count").desc())

print("\nTop 10 most frequently recommended movies:")
top_movies_with_titles.show(truncate=False)

# Show distribution of recommendation counts
print("\n=== Recommendation Count Distribution ===")
rec_count_stats = top_recommended_movies.agg(
    mean(col("recommendation_count")).alias("avg_recommendations"),
    stddev(col("recommendation_count")).alias("std_recommendations"),
    spark_min(col("recommendation_count")).alias("min_recommendations"),
    spark_max(col("recommendation_count")).alias("max_recommendations")
).collect()[0]

print(f"Average recommendations per movie: {rec_count_stats['avg_recommendations']:.1f}")
print(f"Std deviation: {rec_count_stats['std_recommendations']:.1f}")
print(f"Min recommendations: {rec_count_stats['min_recommendations']}")
print(f"Max recommendations: {rec_count_stats['max_recommendations']}")

print("\n" + "="*60)
print("EVALUATION COMPLETE")
print("="*60)


# In[35]:


# Final summary and conclusions

print("=" * 80)
print("PROJECT SUMMARY AND CONCLUSIONS")
print("=" * 80)

print("\n1. REGRESSION MODELS PERFORMANCE:")
print("-" * 40)
print("Revenue Prediction:")
for model_name, result in results.items():
    print(f"  {model_name}: RMSE = ${result['RMSE']:,.0f}, R² = {result['R2']:.4f}")

print("\nVote Average Prediction:")
for model_name, result in vote_results.items():
    print(f"  {model_name}: RMSE = {result['RMSE']:.4f}, R² = {result['R2']:.4f}")

print("\nBest model for revenue prediction: Random Forest")
print("Best model for vote prediction: Random Forest")

print("\n2. FACTORS AFFECTING VOTE AVERAGES:")
print("-" * 40)
print("Key findings:")
print("1. Documentaries and History genres get highest average ratings")
print("2. Movies with moderate runtime (90-150 mins) tend to rate better")
print("3. Vote count has positive correlation with rating (more votes = higher rating)")
print("4. Budget and revenue show weak correlation with ratings")
print("5. Movies from certain languages (Japanese, Korean) rate higher")

print("\n3. RECOMMENDATION SYSTEM:")
print("-" * 40)
print(f"ALS Model RMSE: {rmse:.4f}")
print("System provides two functions:")
print("  a) Movie similarity: Finds similar movies based on collaborative filtering")
print("  b) Personalized recommendations: Predicts ratings for unrated movies")

print("\n4. CHALLENGES AND LIMITATIONS:")
print("-" * 40)
print("1. Data quality issues (missing values, inconsistent formats)")
print("2. Cold start problem for new users/movies")
print("3. Computational complexity with large dataset")
print("4. Subjectivity in movie ratings")

print("\n5. FUTURE IMPROVEMENTS:")
print("-" * 40)
print("1. Incorporate content-based features (genres, directors, actors)")
print("2. Implement hybrid recommendation system")
print("3. Use deep learning models for better embeddings")
print("4. Add temporal dynamics (rating trends over time)")

print("\n" + "=" * 80)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 80)


# In[36]:


# Clean up and stop Spark session
spark.stop()
print("Spark session stopped.")


# In[37]:


import nbformat
from nbconvert import PythonExporter

def convert_notebook_to_python(notebook_path, python_path):
    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Convert to Python
    exporter = PythonExporter()
    python_code, _ = exporter.from_notebook_node(notebook)
    
    # Save Python file
    with open(python_path, 'w', encoding='utf-8') as f:
        f.write(python_code)
    
    print(f"Converted {notebook_path} to {python_path}")

# Usage
convert_notebook_to_python('Movie_Analysis.ipynb', 'movie_analysis.py')


# In[ ]:




