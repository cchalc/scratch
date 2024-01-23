# Databricks notebook source
# MAGIC %pip install deap

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, StructField, DoubleType
from deap import base, creator, tools, algorithms
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
from sklearn.model_selection import train_test_split

# Initialize Spark session
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
spark_df = spark_df.withColumnRenamed("feature_0", "target")
# Show the Spark DataFrame
display(spark_df)

# COMMAND ----------

# Feature Selection with Genetic Algorithm
LABEL_COLUMN = spark_df.columns[0]
NUM_FEATURES = spark_df.columns[1:]  # Assuming label in first column
POPULATION_SIZE = 20
NUM_GENERATIONS = 10
CX_PROB = 0.7
MUT_PROB = 0.2

# COMMAND ----------

# Your custom fitness function (replace with your logic)
def your_fitness_function(individual):
    # Here, you can define the fitness function that evaluates the performance of a given feature set
    # You may train a model and evaluate it based on your specific criteria
    # For example, you can use LightGBM or any other regression model
    selected_feature_columns = [col for col, include in zip(NUM_FEATURES, individual) if include == 1]

    # Train a LightGBM model and evaluate it on your data
    train_data_pd, valid_data_pd = train_test_split(spark_df.toPandas(), test_size=0.2, random_state=42)
    train_dataset = lgb.Dataset(train_data_pd[selected_feature_columns], label=train_data_pd[LABEL_COLUMN])
    valid_dataset = lgb.Dataset(valid_data_pd[selected_feature_columns], reference=train_dataset)

    lgb_params = {
        "objective": "regression",
        "boosting_type": "gbdt",
        "metric": "l2",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9
    }

    model = lgb.train(lgb_params, train_dataset, num_boost_round=100)
    predictions = model.predict(valid_data_pd[selected_feature_columns])
    fitness_score = your_custom_fitness_score(predictions, valid_data_pd[LABEL_COLUMN])

    return (fitness_score,)

# COMMAND ----------



creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Define the genetic algorithm toolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.choice, [0, 1], size=len(NUM_FEATURES))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Genetic Algorithm Configuration
toolbox.register("evaluate", your_fitness_function)  # Replace with your custom fitness function
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Genetic Algorithm Optimization
population = toolbox.population(n=POPULATION_SIZE)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

algorithms.eaSimple(population, toolbox, cxpb=CX_PROB, mutpb=MUT_PROB, ngen=NUM_GENERATIONS, stats=stats, halloffame=hof)
best_feature_set = hof[0]

# COMMAND ----------



# Distribute model training with groupBy().applyInPandas()
@pandas_udf(StructType([StructField("prediction", DoubleType())]))
def train_and_evaluate_lightgbm_udf(data_pd):
    train_data_pd, valid_data_pd = train_test_split(data_pd, test_size=0.2, random_state=42)

    train_dataset = lgb.Dataset(train_data_pd[NUM_FEATURES], label=train_data_pd['target'])
    valid_dataset = lgb.Dataset(valid_data_pd[NUM_FEATURES], reference=train_dataset)

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
    predictions = model.predict(valid_data_pd[NUM_FEATURES])
    return predictions

# COMMAND ----------

# Apply the pandas_udf for each feature set iteration
best_models = []
for feature_set in best_feature_sets:
    selected_feature_columns = [col for col, include in zip(NUM_FEATURES, feature_set) if include == 1]
    result_df = data.select(*selected_feature_columns, 'label').groupBy().apply(train_and_evaluate_lightgbm_udf)
    best_models.append((result_df, feature_set))

# Track the experiments with MLflow
for i, (result_df, feature_set) in enumerate(best_models):
    with mlflow.start_run():
        mlflow.log_params(lgb_params)
        mlflow.lightgbm.log_model(model, "model")
        mlflow.log_metric("best_fitness_score", train_and_evaluate_lightgbm(feature_set))
        mlflow.log_params({
            "num_features_selected": sum(feature_set),
            "population_size": POPULATION_SIZE,
            "num_generations": NUM_GENERATIONS,
            "crossover_probability": CX_PROB,
            "mutation_probability": MUT_PROB,
            "iteration": i
        })
