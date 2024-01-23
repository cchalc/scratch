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
# spark = SparkSession.builder.appName("LightGBM basic").getOrCreate()

# Load and preprocess your data as needed
# Replace this with your own dataset
# For demonstration purposes, we'll create a sample Spark DataFrame
from pyspark.sql import Row
data = spark.createDataFrame([Row(target=3, feature_0=1, feature_1=2, feature_2=3),
                              Row(target=4, feature_0=2, feature_1=3, feature_2=4),
                              Row(target=5, feature_0=3, feature_1=4, feature_2=5),
                              Row(target=6, feature_0=4, feature_1=5, feature_2=6)])

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, StructField, DoubleType
from deap import base, creator, tools, algorithms
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
from sklearn.model_selection import train_test_split

# Define your Spark session (uncomment the following line if not already defined)
# spark = SparkSession.builder.appName("LightGBM basic").getOrCreate()

# Create a sample Spark DataFrame for demonstration
data = spark.createDataFrame([
    (3, 1, 2, 3),
    (4, 2, 3, 4),
    (5, 3, 4, 5),
    (6, 4, 5, 6)
], ["target", "feature_0", "feature_1", "feature_2"])

# Initialize best_feature_set outside of your_fitness_function
best_feature_set = None

# Feature Selection with Genetic Algorithm
POPULATION_SIZE = 20
NUM_GENERATIONS = 10
CX_PROB = 0.7
MUT_PROB = 0.2

# Define the number of features and the label column
LABEL_COLUMN = "target"
NUM_FEATURES = data.columns[1:]

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

# Your custom fitness function (replace with your logic)
def your_fitness_function(individual):
    selected_feature_columns = [col for col, include in zip(NUM_FEATURES, individual) if np.any(include == 1)]

    # Train a LightGBM model and evaluate it on your data
    train_data_pd, valid_data_pd = train_test_split(data.toPandas(), test_size=0.2, random_state=42)
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
    # Your fitness score computation (e.g., RMSE, R-squared, etc.)
    fitness_score = your_custom_fitness_score(predictions, valid_data_pd[LABEL_COLUMN])
    return (fitness_score,)

# Genetic Algorithm Optimization
for gen in range(NUM_GENERATIONS):
    offspring = algorithms.varAnd(population, toolbox, cxpb=CX_PROB, mutpb=MUT_PROB)
    offspring = [ind for ind in offspring if len(ind) >= 2]  # Remove individuals with size less than 2
    fits = toolbox.map(toolbox.evaluate, offspring)

    if offspring:  # Check if there are offspring to update best_feature_set
        best_ind = np.argmax(fits)
        best_feature_set = offspring[best_ind]

    population = offspring

# best_feature_set now contains the best feature selection


# COMMAND ----------

# Distribute model training with groupBy().applyInPandas()
@pandas_udf(StructType([StructField("prediction", DoubleType())]))
def train_and_evaluate_lightgbm_udf(data_pd):
    train_data_pd, valid_data_pd = train_test_split(data_pd, test_size=0.2, random_state=42)

    train_dataset = lgb.Dataset(train_data_pd[NUM_FEATURES], label=train_data_pd[LABEL_COLUMN])
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

# Apply the pandas_udf for the best_feature_set
selected_feature_columns = [col for col, include in zip(NUM_FEATURES, best_feature_set) if include == 1]
result_df = data.select(LABEL_COLUMN, *selected_feature_columns).groupBy().apply(train_and_evaluate_lightgbm_udf)

# Track the experiment with MLflow
with mlflow.start_run():
    mlflow.log_params(lgb_params)
    mlflow.lightgbm.log_model(model, "model")
    mlflow.log_metric("best_fitness_score", your_fitness_function(best_feature_set))
    mlflow.log_params({
        "num_features_selected": sum(best_feature_set),
        "population_size": POPULATION_SIZE,
        "num_generations": NUM_GENERATIONS,
        "crossover_probability": CX_PROB,
        "mutation_probability": MUT_PROB
    })

# COMMAND ----------


