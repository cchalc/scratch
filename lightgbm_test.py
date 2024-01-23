# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, StructField, DoubleType
from deap import base, creator, tools, algorithms
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
from sklearn.model_selection import train_test_split

# # Initialize Spark session
# spark = SparkSession.builder.appName("LightGBM with PySpark and MLflow").getOrCreate()

# COMMAND ----------

from sklearn.datasets import make_classification, make_regression
import lightgbm as lgb
import pandas as pd

#X, y = make_classification(n_samples=10000, n_features=10, random_state=42)
X, y = make_regression(n_samples=10000, n_features=10, random_state=42)
X = pd.DataFrame(X)
y = pd.DataFrame(y)
data = lgb.Dataset(X, label=y)
spark_df = spark.createDataFrame(data.data)

# Rename the columns
new_columns = ["feature_" + str(i) for i in range(len(data.data.columns))]
spark_df = spark_df.toDF(*new_columns)

# Show the Spark DataFrame
display(spark_df)

# COMMAND ----------

# Load and preprocess your data as needed
# Replace this with your own dataset
from lightgbm import datasets
data = datasets.get_regression()

# Feature Selection with Genetic Algorithm
NUM_FEATURES = data.data.shape[1]
POPULATION_SIZE = 20
NUM_GENERATIONS = 10
CX_PROB = 0.7
MUT_PROB = 0.2

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# ... (remaining genetic algorithm code) ...

# Now you have best_feature_set from the genetic algorithm

# Convert Spark DataFrame to Pandas DataFrame
data_pd = data.data

# Split the data into training and validation sets using sklearn's train_test_split
train_data_pd, valid_data_pd = train_test_split(data_pd, test_size=0.2, random_state=42)

# Distribute model training with pandas_udf
@pandas_udf(StructType([StructField("prediction", DoubleType())]))
def train_and_evaluate_lightgbm_udf(data_pd):
    # Construct LightGBM Datasets from Pandas DataFrame
    train_dataset = lgb.Dataset(train_data_pd, label=...)
    valid_dataset = lgb.Dataset(valid_data_pd, reference=train_dataset)

    # Train the LightGBM model for regression
    lgb_params = {
        "objective": "regression",
        "boosting_type": "gbdt",
        "metric": "l2",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9
    }

    model = lgb.train(lgb_params, train_dataset, num_boost_round=100)

    # Make predictions on the validation data
    predictions = model.predict(valid_data_pd)
    return predictions

# Select the feature columns based on best_feature_set
selected_feature_columns = [str(i) for i, include in enumerate(best_feature_set) if include == 1]

# Apply the pandas_udf to perform distributed training and predictions
result_df = data.select(*selected_feature_columns).withColumn("prediction", train_and_evaluate_lightgbm_udf(*selected_feature_columns))

# Track the experiment with MLflow
with mlflow.start_run():
    mlflow.log_params(lgb_params)
    mlflow.lightgbm.log_model(model, "model")
    mlflow.log_metric("best_fitness_score", train_and_evaluate_lightgbm(best_feature_set))
    mlflow.log_params({
        "num_features_selected": sum(best_feature_set),
        "population_size": POPULATION_SIZE,
        "num_generations": NUM_GENERATIONS,
        "crossover_probability": CX_PROB,
        "mutation_probability": MUT_PROB
    })

# View your experiment in MLflow
mlflow.end_run()

# Stop the Spark session
spark.stop()
