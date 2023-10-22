# Databricks notebook source
spark.sql("use catalog cjc")
spark.sql("use schema scratch")

# COMMAND ----------

import pandas as pd

# COMMAND ----------

df = spark.read.table("bbc_news_train")
display(df)

# COMMAND ----------

df.select('Category').distinct().show()

# COMMAND ----------

from transformers import AutoModelForSequenceClassification, AutoTokenizer
nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')

# COMMAND ----------

tokenizer.max_model_input_sizes

# COMMAND ----------

# Convert the Spark DataFrame column to a list
text_list = df.select('Text').rdd.flatMap(lambda x: x).collect()

# Tokenize the list of strings
tokenized_articles_lengths = pd.DataFrame({'length': list(map(len, tokenizer(text_list, truncation=False, padding=False)['input_ids']))})

# Print summary statistics of the resulting DataFrame
tokenized_articles_lengths.describe()

# COMMAND ----------

labels = df.select('Category').distinct().rdd.flatMap(lambda x: x).collect()
# print(labels)
# ['sport', 'politics', 'entertainment', 'business', 'tech']

# COMMAND ----------


