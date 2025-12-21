import matplotlib.pyplot as plt

def analyze_votes(movies):

    print("Correlation with Vote Average:")
    print("Vote Count:",
          movies.stat.corr("vote_average", "vote_count"))
    print("Popularity:",
          movies.stat.corr("vote_average", "popularity"))
    print("Runtime:",
          movies.stat.corr("vote_average", "runtime"))

    sample = movies.select(
        "vote_average", "vote_count"
    ).sample(0.01).toPandas()

    plt.scatter(sample["vote_count"], sample["vote_average"], alpha=0.3)
    plt.xlabel("Vote Count")
    plt.ylabel("Vote Average")
    plt.title("Vote Count vs Vote Average")
    plt.show()
