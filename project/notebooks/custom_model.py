# Databricks notebook source
# MAGIC %md Custom model logging investigation
# MAGIC Source taken from:
# MAGIC - https://mlflow.org/docs/latest/models.html#example-saving-an-xgboost-model-in-mlflow-format

# COMMAND ----------

import sys
print("\n".join(sys.path))

# COMMAND ----------

# Check to see if module loads correctly
from project.code.src.prep_data import load_data

# COMMAND ----------

# Load training and test datasets
from sys import version_info
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split

PYTHON_VERSION = "{major}.{minor}.{micro}".format(
    major=version_info.major, minor=version_info.minor, micro=version_info.micro
)
# iris = datasets.load_iris()
# x = iris.data[:, 2:]
# y = iris.target
x, y = load_data()
x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(x_train, label=y_train)

# Train and save an XGBoost model
xgb_model = xgb.train(params={"max_depth": 10}, dtrain=dtrain, num_boost_round=10)
xgb_model_path = "xgb_model.pth"
xgb_model.save_model(xgb_model_path)

# Create an `artifacts` dictionary that assigns a unique name to the saved XGBoost model file.
# This dictionary will be passed to `mlflow.pyfunc.save_model`, which will copy the model file
# into the new MLflow Model's directory.
artifacts = {"xgb_model": xgb_model_path}

# COMMAND ----------



# Define the model class
import mlflow.pyfunc


class XGBWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import xgboost as xgb

        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(context.artifacts["xgb_model"])

    def predict(self, context, model_input):
        input_matrix = xgb.DMatrix(model_input.values)
        return self.xgb_model.predict(input_matrix)


# Create a Conda environment for the new MLflow Model that contains all necessary dependencies.
import cloudpickle

conda_env = {
    "channels": ["defaults"],
    "dependencies": [
        "python={}".format(PYTHON_VERSION),
        "pip",
        {
            "pip": [
                "mlflow=={}".format(mlflow.__version__),
                "xgboost=={}".format(xgb.__version__),
                "cloudpickle=={}".format(cloudpickle.__version__),
            ],
        },
    ],
    "name": "xgb_env",
}


# COMMAND ----------

# create code_path
import os
import importlib
mod = importlib.import_module('project.code.src.prep_data')
code_path1 = str(mod).split("'")[3]
print(code_path1)

# COMMAND ----------


# Save the MLflow Model
with mlflow.start_run():
  mlflow_pyfunc_model_path = "xgb_mlflow_pyfunc_new"
  mlflow.pyfunc.log_model(
      # path=mlflow_pyfunc_model_path, # used for save_model
      artifact_path="artifacts1",
      python_model=XGBWrapper(),
      # artifacts="artifacts1", # used for save_model
      code_path=[code_path1],
      conda_env=conda_env,
  )



# COMMAND ----------

# Load the model in `python_function` format
loaded_model = mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)

# Evaluate the model
import pandas as pd

test_predictions = loaded_model.predict(pd.DataFrame(x_test))
print(test_predictions)

# COMMAND ----------

metadata_json = loaded_model.metadata.to_json()
display(metadata_json)

# COMMAND ----------


