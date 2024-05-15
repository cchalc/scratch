# Databricks notebook source
# MAGIC %md
# MAGIC # Overview of bert models in Databricks Marketplace Listing
# MAGIC
# MAGIC The bert models offered in Databricks Marketplace are text generation models released by Google. They are [MLflow](https://mlflow.org/docs/latest/index.html) models that packages
# MAGIC [Hugging Face’s implementation for bert models](https://huggingface.co/models?pipeline_tag=fill-mask&sort=trending&search=bert)
# MAGIC using the [transformers](https://mlflow.org/docs/latest/models.html#transformers-transformers-experimental)
# MAGIC flavor in MLflow.
# MAGIC
# MAGIC **Input:** string containing the text with the mask token
# MAGIC
# MAGIC **Output:** string containing the generated text
# MAGIC
# MAGIC For example notebooks of using the bert model in various use cases on Databricks, please refer to [the Databricks ML example repository](https://github.com/databricks/databricks-ml-examples/tree/master/llm-models/bert).

# COMMAND ----------

# MAGIC %md
# MAGIC # Listed Marketplace Models
# MAGIC - bert_base_cased:
# MAGIC   - It packages Hugging Face’s implementation for [bert_base_cased](https://huggingface.co/bert-base-cased).
# MAGIC   - It has 109 Million parameters.
# MAGIC   - It is a pretrained model on English language using a masked language modeling (MLM) objective. This model is case-sensitive: it makes a difference between english and English.
# MAGIC - bert_base_multilingual_cased:
# MAGIC   - It packages Hugging Face’s implementation for [bert_base_multilingual_cased](https://huggingface.co/bert-base-multilingual-cased).
# MAGIC   - It has 179 Million parameters.
# MAGIC   - It is a pretrained model on the top 104 languages using a masked language modeling (MLM) objective. This model is case-sensitive: it makes a difference between english and English.
# MAGIC - bert_base_uncased:
# MAGIC   - It packages Hugging Face’s implementation for [bert_base_uncased](https://huggingface.co/bert-base-uncased).
# MAGIC   - It has 110 Million parameters.
# MAGIC   - It is a pretrained model on English language using a masked language modeling (MLM) objective. This model is uncased: it does not make a difference between english and English.
# MAGIC - bert_large_cased:
# MAGIC   - It packages Hugging Face’s implementation for [bert_large_cased](https://huggingface.co/bert-large-cased).
# MAGIC   - It has 335 Million parameters.
# MAGIC   - It is a pretrained model on English language using a masked language modeling (MLM) objective. This model is case-sensitive: it makes a difference between english and English.
# MAGIC - bert_large_cased_whole_word_masking:
# MAGIC   - It packages Hugging Face’s implementation for [bert_large_cased_whole_word_masking](https://huggingface.co/bert-large-cased-whole-word-masking).
# MAGIC   - It has 336 Million parameters.
# MAGIC   - It is a pretrained model on English language with a new technique: Whole Word Masking. This model is case-sensitive: it makes a difference between english and English.
# MAGIC - bert_large_uncased:
# MAGIC   - It packages Hugging Face’s implementation for [bert_large_uncased](https://huggingface.co/bert-large-uncased).
# MAGIC   - It has 336 Million parameters.
# MAGIC   - It is a pretrained model on English language using a masked language modeling (MLM) objective. This model is uncased: it does not make a difference between english and English.
# MAGIC - bert_large_uncased_whole_word_masking:
# MAGIC   - It packages Hugging Face’s implementation for [bert_large_uncased_whole_word_masking](https://huggingface.co/bert-large-uncased-whole-word-masking).
# MAGIC   - It has 336 Million parameters.
# MAGIC   - It is a pretrained model on English language with a new technique: Whole Word Masking. This model is uncased: it does not make a difference between english and English.

# COMMAND ----------

# MAGIC %md
# MAGIC # Install Dependencies
# MAGIC To create and query the model serving endpoint, Databricks recommends to install the newest Databricks SDK for Python.

# COMMAND ----------

# Upgrade to use the newest Databricks SDK
%pip install --upgrade databricks-sdk
dbutils.library.restartPython()

# COMMAND ----------

# Select the model from the dropdown list
model_names = ['bert_base_cased', 'bert_base_multilingual_cased', 'bert_base_uncased', 'bert_large_cased', 'bert_large_cased_whole_word_masking', 'bert_large_uncased', 'bert_large_uncased_whole_word_masking']
dbutils.widgets.dropdown("model_name", model_names[0], model_names)

# COMMAND ----------

# Default catalog name when installing the model from Databricks Marketplace.
# Replace with the name of the catalog containing this model
# You can also specify a different model version to load for inference
catalog_name = "databricks_bert_models"
version = "1"
model_name = dbutils.widgets.get("model_name")
model_uc_path = f"{catalog_name}.models.{model_name}"
endpoint_name = f'{model_name}_marketplace'
workload_type = "GPU_SMALL"

# COMMAND ----------

# MAGIC %md
# MAGIC # Usage

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks recommends that you primarily work with this model via Model Serving
# MAGIC ([AWS](https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/create-manage-serving-endpoints)).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploying the model to Model Serving
# MAGIC
# MAGIC You can deploy this model directly to a Databricks Model Serving Endpoint
# MAGIC ([AWS](https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/create-manage-serving-endpoints)).
# MAGIC
# MAGIC Note: Model serving is not supported on GCP. On GCP, Databricks recommends running `Batch inference using Spark`, 
# MAGIC as shown below.
# MAGIC
# MAGIC We recommend the below workload types for each model size:
# MAGIC | Model Name      | Suggested workload type (AWS) | Suggested workload type (AZURE) |
# MAGIC | --------------- | ----------------------------- | ------------------------------- |
# MAGIC | `bert_base_cased` | GPU_SMALL | GPU_SMALL  |
# MAGIC | `bert_base_multilingual_cased` | GPU_SMALL | GPU_SMALL  |
# MAGIC | `bert_base_uncased` | GPU_SMALL | GPU_SMALL  |
# MAGIC | `bert_large_cased` | GPU_SMALL | GPU_SMALL  |
# MAGIC | `bert_large_cased_whole_word_masking` | GPU_SMALL | GPU_SMALL  |
# MAGIC | `bert_large_uncased` | GPU_SMALL | GPU_SMALL  |
# MAGIC | `bert_large_uncased_whole_word_masking` | GPU_SMALL | GPU_SMALL  |
# MAGIC
# MAGIC You can create the endpoint by clicking the “Serve this model” button above in the model UI. And you can also
# MAGIC create the endpoint with Databricks SDK as following:

# COMMAND ----------

import datetime

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput
w = WorkspaceClient()

config = EndpointCoreConfigInput.from_dict({
    "served_models": [
        {
            "name": endpoint_name,
            "model_name": model_uc_path,
            "model_version": version,
            "workload_type": workload_type,
            "workload_size": "Small",
            "scale_to_zero_enabled": "False",
        }
    ]
})
model_details = w.serving_endpoints.create(name=endpoint_name, config=config)
model_details.result(timeout=datetime.timedelta(minutes=40))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate the text by querying the serving endpoint
# MAGIC With the Databricks SDK, you can query the serving endpoint as follows:

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Change it to your own input
dataframe_records = ["Paris is the [MASK] of France."]

response = w.serving_endpoints.query(
    name=endpoint_name,
    dataframe_records=dataframe_records,
)

response.predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch inference using Spark
# MAGIC
# MAGIC You can directly load the model as a Spark UDF and run batch
# MAGIC inference on Databricks compute using Spark. We recommend using a
# MAGIC GPU cluster with Databricks Runtime for Machine Learning version 14.1
# MAGIC or greater.

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

logged_model = f"models:/{catalog_name}.models.{model_name}/{version}"
generate = mlflow.pyfunc.spark_udf(spark, logged_model, "string")

# COMMAND ----------

import pandas as pd

df = spark.createDataFrame(pd.DataFrame({"text": pd.Series("Paris is the [MASK] of France.")}))

# You can use the UDF directly on a text column
generated_df = df.select(generate(df.text).alias('generated_text'))
display(generated_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch inference in Notebook
# MAGIC
# MAGIC You can also load the model and run batch inference

# COMMAND ----------

loaded_model = mlflow.transformers.load_model(logged_model)

data = "Paris is the [MASK] of France."
print(loaded_model.predict(data))
