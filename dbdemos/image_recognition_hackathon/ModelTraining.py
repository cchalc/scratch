# Databricks notebook source
#python Imports for ML...
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn
from mlflow.tracking.client import MlflowClient
from hyperopt import fmin, hp, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from hyperopt import SparkTrials
from sklearn.model_selection import GroupKFold
from pyspark.sql.functions import pandas_udf, PandasUDFType
import os
import pandas as pd
from hyperopt import space_eval
import numpy as np
from time import sleep

# COMMAND ----------

def init_experiment_for_batch(demo_name, experiment_name):
  #You can programatically get a PAT token with the following
  pat_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
  url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
  headers = {"Accept": "application/json", "Authorization": f"Bearer {pat_token}"}
  import requests
  xp_root_path = f"/Shared/dbdemos/experiments/{demo_name}"
  r = requests.post(f"{url}/api/2.0/workspace/mkdirs", headers = headers, json={ "path": xp_root_path})
  if r.status_code != 200:
    print(f"ERROR: couldn't create a folder for the experiment under {xp_root_path} - please create the folder manually or  skip this init (used for job only: {r})")
  else:
    for i in range(3):
      #Wait to make sure the folder is created cause it's asynch?
      folder = requests.get(f"{url}/api/2.0/workspace/list", headers = headers, params={ "path": xp_root_path}).json()
      if folder.get('error_code', "") != 'RESOURCE_DOES_NOT_EXIST':
        break
    #time.sleep(1*i)
  xp = f"{xp_root_path}/{experiment_name}"
  print(f"Using common experiment under {xp}")
  mlflow.set_experiment(xp)
  #set_experiment_permission(xp)
  return mlflow.get_experiment_by_name(xp)

# COMMAND ----------

#Setup the training experiment
init_experiment_for_batch("garbage-computer-vision-dl", "waste")

df = spark.read.table("garbage_training_dataset")
display(df.limit(10))

# COMMAND ----------

#Create the transformer dataset from a spark dataframe (Delta Table)  
from datasets import Dataset

dataset = Dataset.from_spark(df).rename_column("content", "image")

splits = dataset.train_test_split(test_size=0.2, seed = 42)
train_ds = splits['train']
val_ds = splits['test']

# COMMAND ----------

import torch
from transformers import AutoFeatureExtractor, AutoImageProcessor

# pre-trained model from which to fine-tune
# Check the hugging face repo for more details & models: https://huggingface.co/google/vit-base-patch16-224
model_checkpoint = "google/vit-base-patch16-224"

#Check GPU availability
if not torch.cuda.is_available(): # is gpu
  raise Exception("Please use a GPU-cluster for model training, CPU instances will be too slow")

# COMMAND ----------

from PIL import Image
import io
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomResizedCrop, Resize, ToTensor, Lambda

#Extract the model feature (contains info on pre-process step required to transform our data, such as resizing & normalization)
#Using the model parameters makes it easy to switch to another model without any change, even if the input size is different.
model_def = AutoFeatureExtractor.from_pretrained(model_checkpoint)

normalize = Normalize(mean=model_def.image_mean, std=model_def.image_std)
byte_to_pil = Lambda(lambda b: Image.open(io.BytesIO(b)).convert("RGB"))

#Transformations on our training dataset. we'll add some crop here
train_transforms = Compose([byte_to_pil,
                            RandomResizedCrop((model_def.size['height'], model_def.size['width'])),
                            ToTensor(), #convert the PIL img to a tensor
                            normalize
                           ])
#Validation transformation, we only resize the images to the expected size
val_transforms = Compose([byte_to_pil,
                          Resize((model_def.size['height'], model_def.size['width'])),
                          ToTensor(),  #convert the PIL img to a tensor
                          normalize
                         ])

# Add some random resiz & transformation to our training dataset
def preprocess_train(batch):
    """Apply train_transforms across a batch."""
    batch["image"] = [train_transforms(image) for image in batch["image"]]
    return batch

# Validation dataset
def preprocess_val(batch):
    """Apply val_transforms across a batch."""
    batch["image"] = [val_transforms(image) for image in batch["image"]]
    return batch
  
#Set our training / validation transformations
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

# COMMAND ----------

from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

#Mapping between class label and value (huggingface use it during inference to output the proper label)
label2id, id2label = dict(), dict()
for i, label in enumerate(set(dataset['labels'])):
    label2id[label] = i
    id2label[i] = label
    
model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint, 
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)

# COMMAND ----------

model_name = model_checkpoint.split("/")[-1]
batch_size = 32 # batch size for training and evaluation

args = TrainingArguments(
    f"/tmp/huggingface/pcb/{model_name}-finetuned-leaf",
    remove_unused_columns=False,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=20,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False
)

# COMMAND ----------

import numpy as np
import evaluate
# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.

# Let's evaluate our model against a F1 score. Keep it as binary for this demo (we don't classify by default type)
accuracy = evaluate.load("f1")

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

# COMMAND ----------

import mlflow
import torch
from transformers import pipeline, DefaultDataCollator, EarlyStoppingCallback

def collate_fn(examples):
    pixel_values = torch.stack([e["image"] for e in examples])
    labels = torch.tensor([label2id[e["labels"]] for e in examples])
    return {"pixel_values": pixel_values, "labels": labels}

#Make sure the model is trained on GPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

#mlflow.autolog(log_models=False)
with mlflow.start_run(run_name="hugging_face") as run:
  early_stop = EarlyStoppingCallback(early_stopping_patience=10)
  trainer = Trainer(model, args, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=model_def, compute_metrics=compute_metrics, data_collator=collate_fn, callbacks = [early_stop])

  train_results = trainer.train()

  #Build our final hugging face pipeline
  classifier = pipeline("image-classification", model=trainer.state.best_model_checkpoint, tokenizer = model_def, device_map='auto')
  #log the model to MLFlow
  reqs = mlflow.transformers.get_default_pip_requirements(model)
  mlflow.transformers.log_model(artifact_path="model", transformers_model=classifier, pip_requirements=reqs)
  mlflow.set_tag("dbdemos", "waste_classification")
  mlflow.log_metrics(train_results.metrics)

# COMMAND ----------

test = spark.read.table("garbage_training_dataset").where("path = 'dbfs:/mnt/images/AbGarbageImages/2019920.jpg'").toPandas()
img = Image.open(io.BytesIO(test.iloc[0]['content']))
print(f"predictions: {classifier(img)}")
display(img)
