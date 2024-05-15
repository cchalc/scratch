# Databricks notebook source
# MAGIC %md
# MAGIC ### pip installs

# COMMAND ----------

# MAGIC %pip install mlflow==2.9.0 lxml==4.9.3 langchain==0.0.344 databricks-vectorsearch==0.22 cloudpickle==2.2.1 databricks-sdk==0.12.0 cloudpickle==2.2.1 pydantic==2.5.2

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup

# COMMAND ----------

dbutils.widgets.text("catalog_name","cjc")

# COMMAND ----------

user_name = spark.sql("SELECT current_user() as username").collect()[0].username
schema_name = 'scratch'
catalog_name = dbutils.widgets.get('catalog_name')
cache_name = 'llm_cache'

spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
spark.sql(f"USE SCHEMA {schema_name}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {cache_name}")

# COMMAND ----------

import os
import pandas as pd
import transformers as tr
from datasets import load_dataset

# COMMAND ----------

# ai classify found here: https://e2-demo-field-eng.cloud.databricks.com/sql/editor/0821532c-a5df-4e51-a8b2-e112f40f8b96?o=1444828305810485
tables = spark.sql(f"SHOW TABLES IN {schema_name}").toPandas()
tables

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from cjc.scratch.bbc_news_train limit 10

# COMMAND ----------

# MAGIC %md
# MAGIC ### AI Functions
# MAGIC [ai_classify](https://docs.databricks.com/en/sql/language-manual/functions/ai_classify.html) found here: https://e2-demo-field-eng.cloud.databricks.com/sql/editor/0821532c-a5df-4e51-a8b2-e112f40f8b96?o=1444828305810485
# MAGIC

# COMMAND ----------

# MAGIC %md ### Prompt Engineering with LangChain

# COMMAND ----------

import langchain
print(langchain.__version__)

# COMMAND ----------

import os
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chat_models import ChatDatabricks

#call llama2 70B, hosted by Databricks
llama_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 400)

# COMMAND ----------

from langchain import PromptTemplate
from langchain.chains import LLMChain

def run_llm_chain(input_string, template_string, model):
  """
  given an input string, template, and model, execute a langchain chain on the input with a given prompt template

  params
  ==========
  input_string (str): the incoming query from a user to be evaluated
  template_string (str): prompt template append or pre-pend to input_string (required for prompt engineering)
  model (langchain model): the name of the model 
  """
  prompt_template = PromptTemplate(
    input_variables=["input_string"],
    template=template_string,
  )
  model_chain = LLMChain(
    llm=model,
    prompt=prompt_template,
    output_key="Response",
    verbose=False
  )

  return model_chain.run({"input_string": input_string})

# COMMAND ----------

zero_shot_template = """For each block of text, classify into one of the following categories: ["tech","business","entertainment","sport","politics"] and only return the category without a description or any other text.
                        [Text]: {input_string}
                      """

# COMMAND ----------

input_string = """
worldcom ex-boss launches defence lawyers defending former worldcom chief bernie ebbers against a battery of fraud charges have called a company whistleblower as their first witness.  cynthia cooper  worldcom s ex-head of internal accounting  alerted directors to irregular accounting practices at the us telecoms giant in 2002. her warnings led to the collapse of the firm following the discovery of an $11bn (£5.7bn) accounting fraud. mr ebbers has pleaded not guilty to charges of fraud and conspiracy.  prosecution lawyers have argued that mr ebbers orchestrated a series of accounting tricks at worldcom  ordering employees to hide expenses and inflate revenues to meet wall street earnings estimates. but ms cooper  who now runs her own consulting business  told a jury in new york on wednesday that external auditors arthur andersen had approved worldcom s accounting in early 2001 and 2002. she said andersen had given a  green light  to the procedures and practices used by worldcom. mr ebber s lawyers have said he was unaware of the fraud  arguing that auditors did not alert him to any problems.  ms cooper also said that during shareholder meetings mr ebbers often passed over technical questions to the company s finance chief  giving only  brief  answers himself. the prosecution s star witness  former worldcom financial chief scott sullivan  has said that mr ebbers ordered accounting adjustments at the firm  telling him to  hit our books . however  ms cooper said mr sullivan had not mentioned  anything uncomfortable  about worldcom s accounting during a 2001 audit committee meeting. mr ebbers could face a jail sentence of 85 years if convicted of all the charges he is facing. worldcom emerged from bankruptcy protection in 2004  and is now known as mci. last week  mci agreed to a buyout by verizon communications in a deal valued at $6.75bn.
"""
zero_shot_response = run_llm_chain(input_string, zero_shot_template, llama_model)
print(zero_shot_response)

# COMMAND ----------

# MAGIC %md ### Classify with BART

# COMMAND ----------

os.environ['TRANSFORMERS_CACHE'] = '/Volumes/cjc/scratch/llm_cache'

# COMMAND ----------

df = spark.read.table("cjc.scratch.bbc_news_train")
df.show(2)

# COMMAND ----------

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
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

candidate_labels = df.select('Category').distinct().rdd.flatMap(lambda x: x).collect()


# COMMAND ----------

# Use GPU if available
# device = 0 if torch.cuda.is_available() else -1

# Define the zero-shot classification model
pipe = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device_map='auto',
    truncation=True
)

# COMMAND ----------

pipe(input_string, candidate_labels)

# COMMAND ----------

# MAGIC %md #### Batch Classify

# COMMAND ----------

subset = df.limit(40)
pdf = subset.toPandas()
pdf.head()

# COMMAND ----------

from pyspark.sql.functions import pandas_udf

broadcast_pipeline = spark.sparkContext.broadcast(zero_shot_classifier)

# Define a Pandas UDF to perform zero-shot classification
@pandas_udf('string')
def classify_text(texts: pd.Series) -> pd.Series:
    pipe = broadcast_pipeline.value(
        ("classify: " + texts).to_list(),
        candidate_labels,
        batch_size=8
    )
    labels = [result['labels'][0] for result in pipe]
    return pd.Series(labels)

# COMMAND ----------

sc.setJobDescription("single classification")
result_df = (df
             .limit(40)
             .cache()
             .withColumn(
               "label_pred_zero_shot", classify_text(df.Text)
               )
             )

# COMMAND ----------

display(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logging and Registering with MLflow
# MAGIC <img src="https://miro.medium.com/v2/resize:fit:1400/1*OVqzvRSNWloHMYCF1EZtqg.png" alt="mlflow" width="400"/>
# MAGIC
# MAGIC Now that we have our model, we want to log the model and its artifacts, so we can version it, deploy it, and also share it with other users.

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature
from mlflow.transformers import generate_signature_output
"""
For LLMs, we need to generate a model signature: https://mlflow.org/docs/latest/models.html#model-signature-and-input-example
Model signatures show the expected input and output types for a model. Which makes quality assurance for downstream serving easier
"""

# COMMAND ----------

data = input_string
output = generate_signature_output(pipe, data)
signature = infer_signature(data, output)

# COMMAND ----------

# MAGIC %md
# MAGIC #### references
# MAGIC - [Rapid intro to transformers](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485&name-order=ascend#notebook/4145428257205067/command/648780469062853)
# MAGIC - [Prompt Engineering](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#notebook/4145428257205068/command/4145428257205232)
# MAGIC - [Fine Tuning](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#notebook/648780469140250/command/648780469140264)

# COMMAND ----------


