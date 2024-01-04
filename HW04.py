from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql.functions import col, avg, desc, lit, sqrt, udf
from pyspark.sql.types import FloatType
from collections import defaultdict
from pyspark.ml.recommendation import ALS
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.ml.linalg import Vectors, VectorUDT
import itertools
import numpy as np
import pandas as pd


def main():
    spark = (
        SparkSession.builder.appName("MovieLens Recommender System")
        .master("spark://96.9.210.170:7077")
        .config("spark.hadoop.validateOutputSpecs", "false")
        .config(
            "spark.hadoop.home.dir",
            "C:/Users/Asus/Downloads/spark-3.5.0-bin-hadoop3/spark-3.5.0-bin-hadoop3/bin",
        )
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "14g")
        .config("spark.memory.fraction", "0.8")
        .config("spark.memory.storageFraction", "0.5")
        .config(
            "spark.executor.extraJavaOptions", "-XX:+UseG1GC"
        )  # Using G1 Garbage Collector
        .config(
            "spark.driver.extraJavaOptions", "-XX:+UseG1GC"
        )  # Using G1 Garbage Collector for driver
        .getOrCreate()
    )
    sc = spark.sparkContext

    # Read in the data files
    movies = (
        spark.read.option("delimiter", "::")
        .csv("ml-1m/ml-1m/movies.dat", inferSchema=True, header=False)
        .toDF("MovieID", "Title", "Genres")
    )
    ratings = (
        spark.read.option("delimiter", "::")
        .csv("ml-1m/ml-1m/ratings.dat", inferSchema=True, header=False)
        .toDF("UserID", "MovieID", "Rating", "Timestamp")
    )
    users = (
        spark.read.option("delimiter", "::")
        .csv("ml-1m/ml-1m/users.dat", inferSchema=True, header=False)
        .toDF("UserID", "Gender", "Age", "Occupation", "Zip-code")
    )

    # TASK 1: Save the list of <movie, score> pairs in descending order of ‘average’ rating score
    task1 = (
        ratings.groupBy("MovieID")
        .agg({"Rating": "avg"})
        .join(movies, "MovieID")
        .orderBy("avg(Rating)", ascending=False)
    )
    task1_pd = task1.toPandas()
    task1_pd.to_csv("output/task1.csv", index=False)

    # TASK 2: Save the 3 sorted lists of <movie, gender, score>, <movie, age group, score>, <movie, occupation, score>

    # Gender
    task2_gender = (
        ratings.join(users, "UserID")
        .groupBy("MovieID", "Gender")
        .agg({"Rating": "avg"})
        .join(movies, "MovieID")
        .orderBy("Gender", "avg(Rating)", ascending=False)
    )
    task2_gender_pd = task2_gender.toPandas()
    task2_gender_pd.to_csv("output/task2_gender.csv", index=False)

    # Age Group
    task2_age = (
        ratings.join(users, "UserID")
        .groupBy("MovieID", "Age")
        .agg({"Rating": "avg"})
        .join(movies, "MovieID")
        .orderBy("Age", "avg(Rating)", ascending=False)
    )
    task2_age_pd = task2_age.toPandas()
    task2_age_pd.to_csv("output/task2_age.csv", index=False)

    # Occupation
    task2_occupation = (
        ratings.join(users, "UserID")
        .groupBy("MovieID", "Occupation")
        .agg({"Rating": "avg"})
        .join(movies, "MovieID")
        .orderBy("Occupation", "avg(Rating)", ascending=False)
    )
    task2_occupation_pd = task2_occupation.toPandas()
    task2_occupation_pd.to_csv("output/task2_occupation.csv", index=False)

    # TASK 3: Save the two sorted lists of <user, score> pairs, <user, genre, score> pairs

    # User, Score
    task3_user = (
        ratings.groupBy("UserID")
        .agg({"Rating": "avg"})
        .orderBy("avg(Rating)", ascending=False)
    )
    task3_user_pd = task3_user.toPandas()
    task3_user_pd.to_csv("output/task3_user.csv", index=False)

    # User, Genre, Score - this will require a join between ratings and movies
    task3_user_genre = (
        ratings.join(movies, "MovieID")
        .groupBy("UserID", "Genres")
        .agg({"Rating": "avg"})
        .orderBy("UserID", "avg(Rating)", ascending=False)
    )
    task3_user_genre_pd = task3_user_genre.toPandas()
    task3_user_genre_pd.to_csv("output/task3_user_genre.csv", index=False)

    # TASK 4:

    # Initialize ALS learner
    als = ALS(
        maxIter=5,
        regParam=0.01,
        userCol="UserID",
        itemCol="MovieID",
        ratingCol="Rating",
        coldStartStrategy="drop",
    )
    model = als.fit(ratings)

    userRecs = model.recommendForAllUsers(
        10
    )  # Get top 10 recommendations for all users
    userRecs = userRecs.withColumn("rec_exp", explode("recommendations")).select(
        "UserID", "rec_exp.MovieID", "rec_exp.rating"
    )
    userRecs.show()

    # TASK 5

    movieRecs = model.recommendForAllItems(
        10
    )  # Get top 10 recommendations for all items
    movieRecs = movieRecs.withColumn("rec_exp", explode("recommendations")).select(
        "MovieID", "rec_exp.UserID", "rec_exp.rating"
    )
    movieRecs.show()

    # Stop Spark session
    spark.stop()


if __name__ == "__main__":
    main()
