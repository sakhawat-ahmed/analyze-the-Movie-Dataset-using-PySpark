# Movie Dataset Analysis with PySpark

## 📊 Project Overview
A comprehensive movie analysis and recommendation system using PySpark, analyzing 45,000+ movies from the Full MovieLens Dataset. This project implements regression models, data visualization, and collaborative filtering for movie recommendations.

## 🎯 Project Objectives
1. **Regression Analysis**: Predict movie revenue and vote averages using machine learning models
2. **Exploratory Analysis**: Identify factors contributing to higher movie ratings
3. **Recommendation System**: Build collaborative filtering-based movie recommendations

## 📁 Dataset Description
The project uses five datasets from MovieLens:
- **movies_metadata.csv**: Main metadata (45,000 movies)
- **keywords.csv**: Movie plot keywords
- **credits.csv**: Cast and crew information
- **links.csv**: TMDB and IMDB ID mappings
- **ratings.csv**: 26 million ratings from 270,000 users

## 🛠️ Technologies Used
- **PySpark**: Big data processing and ML
- **Python**: Data analysis and visualization
- **MLlib**: Machine learning algorithms
- **Plotly/Matplotlib**: Data visualization
- **Pandas**: Data manipulation

## 📋 Project Structure

### 1. Data Preprocessing
- Data cleaning and transformation
- Handling missing values and type conversions
- Feature engineering (profitability, genre lists, etc.)

### 2. Regression Models
**Revenue Prediction:**
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

**Vote Average Prediction:**
- Same models adapted for rating prediction
- Evaluation using RMSE and R² metrics

### 3. Exploratory Data Analysis
- Distribution analysis of revenue and ratings
- Correlation studies between budget, revenue, and ratings
- Genre-based analysis of movie performance
- Temporal analysis of movie trends

### 4. Recommendation System
**Collaborative Filtering with ALS:**
- Alternating Least Squares implementation
- Two functionality modes:
  - **Movie Similarity**: Find movies similar to a given title
  - **Personalized Recommendations**: Predict ratings for unrated movies

## 📈 Key Features

### ✅ Implemented Features
- **Multi-model regression analysis** with comparative evaluation
- **Comprehensive EDA** with interactive visualizations
- **Scalable recommendation system** using PySpark ALS
- **Data quality checks** and preprocessing pipelines
- **Model performance evaluation** with multiple metrics

### 📊 Analysis Results
1. **Revenue Prediction**: Random Forest performed best with lowest RMSE
2. **Rating Factors**: Vote count, runtime, and budget show strongest correlations
3. **Recommendation Quality**: Achieved competitive RMSE on test data

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Java 11 or 17
- 8GB+ RAM recommended

### Installation Steps
```bash
# Clone repository
git clone <repository-url>
cd movie-analysis-pyspark

# Install dependencies
pip install pyspark==3.5.0
pip install pandas numpy matplotlib seaborn plotly

# Install Java (if not installed)
sudo apt-get install openjdk-11-jdk-headless
```

### Environment Setup
```python
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ["PYSPARK_PYTHON"] = "python3"
```

## 💻 Usage

### Running the Project
```python
# Start Spark session
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .appName("MovieAnalysis") \
    .master("local[*]") \
    .getOrCreate()

# Run main analysis
from movie_analysis import MovieAnalyzer
analyzer = MovieAnalyzer(spark)
analyzer.run_full_analysis()
```

### Key Functions
```python
# 1. Run regression analysis
results = analyzer.run_regression_analysis()

# 2. Generate visualizations
analyzer.create_eda_visualizations()

# 3. Get recommendations
similar_movies = analyzer.get_similar_movies("The Dark Knight", n=10)
user_recs = analyzer.predict_user_ratings(user_id=1, n=10)
```

## 📊 Results & Findings

### Regression Performance
| Model | Revenue RMSE | Revenue R² | Vote RMSE | Vote R² |
|-------|-------------|------------|-----------|---------|
| Linear Regression | $XX,XXX,XXX | 0.XXX | 0.XXX | 0.XXX |
| Random Forest | $XX,XXX,XXX | 0.XXX | 0.XXX | 0.XXX |
| Gradient Boosting | $XX,XXX,XXX | 0.XXX | 0.XXX | 0.XXX |

### Key Insights
1. **Budget-Rating Correlation**: Weak correlation (0.XX)
2. **Vote Count Impact**: Strong positive correlation with ratings
3. **Optimal Runtime**: Movies 90-120 minutes tend to rate better
4. **Genre Performance**: Documentaries and History genres have highest average ratings

## 🧪 Testing & Evaluation

### Model Evaluation Metrics
- **RMSE** (Root Mean Square Error)
- **R²** (Coefficient of Determination)
- **MAE** (Mean Absolute Error)
- **Precision@K** for recommendations

### Recommendation System Evaluation
- Test RMSE: 0.XXXX
- Precision@10: 0.XX
- Cold-start handling implemented

## 📈 Visualization Samples
The project generates comprehensive visualizations including:
- Revenue and rating distributions
- Budget vs revenue scatter plots
- Genre-based performance charts
- Temporal trends in movie metrics

## 🔧 Configuration

### Spark Configuration
```python
spark_config = {
    "driver.memory": "4g",
    "executor.memory": "4g",
    "sql.shuffle.partitions": "50",
    "driver.maxResultSize": "2g"
}
```

### Model Parameters
```python
als_params = {
    "rank": 10,
    "maxIter": 10,
    "regParam": 0.1,
    "coldStartStrategy": "drop"
}

rf_params = {
    "numTrees": 50,
    "maxDepth": 10,
    "seed": 42
}
```

## 🚨 Error Handling
The code includes comprehensive error handling for:
- Missing data files
- Type conversion errors
- Memory constraints
- Model training failures

## 📚 Academic References
This project builds upon research in:
1. Collaborative Filtering (Koren et al., 2009)
2. Movie Success Prediction (Sharda & Delen, 2006)
3. Recommendation Systems (Adomavicius & Tuzhilin, 2005)
4. PySpark MLlib (Meng et al., 2016)

## 🎯 Future Enhancements
1. **Content-based filtering** integration
2. **Deep learning models** for improved accuracy
3. **Real-time recommendation** API
4. **A/B testing framework** for recommendations
5. **Sentiment analysis** of movie reviews

## 📝 Project Report
The complete project report includes:
- Literature review of recommender systems
- Methodology and implementation details
- Results analysis and discussion
- Comparative study with existing approaches
- Limitations and future work


## 📄 License
This project is for educational purposes. Dataset courtesy of GroupLens Research.

## 🙏 Acknowledgments
- MovieLens dataset provided by GroupLens Research
- PySpark community for documentation and support
- Academic papers referenced in the project


---

**Note**: This project requires significant computational resources. For optimal performance, consider using cloud services like Google Colab or AWS EMR for large-scale execution.