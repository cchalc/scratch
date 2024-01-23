# Databricks notebook source
# MAGIC %md ### Setup libraries

# COMMAND ----------

# MAGIC %pip install deap

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

#from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, StructField, DoubleType
from deap import base, creator, tools, algorithms
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# Initialize Spark session
# spark = SparkSession.builder.appName("LightGBM with genetic algorithm").getOrCreate()

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

# Create a pandas dataframe
data_pd = data.data
data_pd.columns = new_columns
data_pd.head()

# COMMAND ----------

label = "feature_9"
model_name = "cjc_lgbm"

# COMMAND ----------

from sklearn.model_selection import train_test_split

@pandas_udf(StructType([StructField("prediction", DoubleType())]))
def train_and_evaluate_lightgbm_udf(data_pd: pd.DataFrame) -> pd.DataFrame:
    train_data, valid_data = train_test_split(data_pd, test_size=0.2, random_state=42)
    train_dataset = lgb.Dataset(train_data, label=label)
    model = lgb.train(lgb_params, train_dataset, num_boost_round=100)
    predictions = model.predict(valid_data)
    return predictions

# COMMAND ----------

# Feature Selection with Genetic Algorithm
NUM_FEATURES = 10
POPULATION_SIZE = 20
NUM_GENERATIONS = 10
CX_PROB = 0.7
MUT_PROB = 0.2

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def eval_feature_set(individual):
    fitness_score = train_and_evaluate_lightgbm(individual)
    return (fitness_score,)

def train_and_evaluate_lightgbm(selected_features):
    train_data, valid_data = train_test_split(data, test_size=0.2, random_state=42)
    selected_feature_columns = [f"feature_{i}" for i, include in enumerate(selected_features) if include == 1]
    train_data = train_data.select(*selected_feature_columns)
    valid_data = valid_data.select(*selected_feature_columns)

    train_dataset = lgb.Dataset(train_data.toPandas(), label=label)
    valid_dataset = lgb.Dataset(valid_data.toPandas(), reference=train_dataset)
    
    lgb_params = {
        "objective": "regression",
        "boosting_type": "gbdt",
        "metric": "l2",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9
    }

    model = lgb.train(lgb_params, train_dataset, valid_sets=[valid_dataset], num_boost_round=100, early_stopping_rounds=10)
    predictions = model.predict(valid_data.toPandas())

    return predictions

# COMMAND ----------

toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.choice, [0, 1])
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=NUM_FEATURES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", eval_feature_set)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=POPULATION_SIZE)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

algorithms.eaSimple(population, toolbox, cxpb=CX_PROB, mutpb=MUT_PROB, ngen=NUM_GENERATIONS, stats=stats, halloffame=hof)

best_feature_set = hof[0]

# # Distribute model training with pandas_udf
# @pandas_udf(StructType([StructField("prediction", DoubleType())]))
# def train_and_evaluate_lightgbm_udf(data):
#     train_data, valid_data = data.randomSplit([0.8, 0.2], seed=42)
#     train_dataset = lgb.Dataset(train_data.toPandas(), label=label)
#     model = lgb.train(lgb_params, train_dataset, num_boost_round=100)
#     predictions = model.predict(valid_data.toPandas())
#     return predictions

# COMMAND ----------

selected_feature_columns = [f"feature_{i}" for i, include in enumerate(best_feature_set)]
result_df = spark_df.select(*selected_feature_columns).withColumn("prediction", train_and_evaluate_lightgbm_udf(*selected_feature_columns))

# Track the experiment with MLflow
with mlflow.start_run():
    mlflow.log_params(lgb_params)
    mlflow.lightgbm.log_model(model, model_name)
    mlflow.log_metric("best_fitness_score", train_and_evaluate_lightgbm(best_feature_set))
    mlflow.log_params({
        "num_features_selected": sum(best_feature_set),
        "population_size": POPULATION_SIZE,
        "num_generations": NUM_GENERATIONS,
        "crossover_probability": CX_PROB,
        "mutation_probability": MUT_PROB
    })

# COMMAND ----------


