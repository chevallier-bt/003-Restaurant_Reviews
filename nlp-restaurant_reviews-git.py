# Databricks notebook source
# MAGIC %md # Restaurant Reviews NLP implementation
# MAGIC This is my first project on DataBricks AWS using PySpark. This is my first NLP problem in ML.
# MAGIC 
# MAGIC ## Objective
# MAGIC Categorize customer sentiment using the provided review. The provided dataset has 

# COMMAND ----------

# MAGIC %md # Familiarizing
# MAGIC There are several practices in NLP that need just a bit of exploration. In order:
# MAGIC * Tokenization
# MAGIC   * Splitting of sentences into individual words for further processing, seems to be one of the first steps in NLP data preparation.
# MAGIC * Stop-Word Removal
# MAGIC   * Removal of unnecessary words which do not provide any contextual meaning to the sentences. Words like 'the', 'of', 'to', etc.
# MAGIC * N-Grams
# MAGIC   * Grouping of individual words into n-long 'sentences'. This can be useful for identifying particular n-grams which can provide important information. Things such as 'tasted bad', 'great price', etc.
# MAGIC * Term Frequency
# MAGIC   * Checking the frequency of particular terms, which may or may not have correlation with outputs. Inverse frequency is also commonly used under the following principle: Words that appear exceedingly often most likely have less unique actionable information than less-frequent words. For exmaple, comparing 'stellar' with 'okay'.

# COMMAND ----------

import pyspark
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, StringType, StructField, StructType, BooleanType, ArrayType, TimestampType

# COMMAND ----------

df = sqlContext.sql("Select * FROM nlp_restaurant_reviews")
df.show(5)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md ## Tokenization
# MAGIC Splitting of sentences into individual words for processing

# COMMAND ----------

from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import udf

# COMMAND ----------

tokenizer = Tokenizer(inputCol='Review', outputCol='words') # Splits sentences into whole words

regexTokenizer = RegexTokenizer(inputCol='Review', outputCol='words', pattern='\\W') # Splits sentences into whole words, ignores punctuation

countTokens = udf(lambda p: len(p), IntegerType()) # udf applies the lambda function to each row when called on a column

# COMMAND ----------

tokenized = tokenizer.transform(df)
tokenized.withColumn('tokens', countTokens(col('words'))).show()

# COMMAND ----------

# MAGIC %md ## Stop Word Removal
# MAGIC 
# MAGIC Removal of common words which add little to no meaning to the sentence

# COMMAND ----------

from pyspark.ml.feature import StopWordsRemover

# COMMAND ----------

df.show(5, truncate=False)

# COMMAND ----------

remover = StopWordsRemover(inputCol='words', outputCol='cleaned')
remover.transform(tokenized).select('cleaned').show(5, truncate=False)

# COMMAND ----------

# MAGIC %md ## n-grams

# COMMAND ----------

from pyspark.ml.feature import NGram

# COMMAND ----------

bigram = NGram(n=2, inputCol='words', outputCol='bigrams')
bigram_df = bigram.transform(tokenized).select('bigrams').show(5, truncate=False)

# COMMAND ----------

# MAGIC %md ## Term Freq; Inverse Doc Freq (TF-IDF)
# MAGIC Importance of specific words. Generally, the more a word is used, the less unique meaning it carries within a corpus. Thus, inverse frequency is used.

# COMMAND ----------

from pyspark.ml.feature import HashingTF, IDF, Tokenizer

# COMMAND ----------

tokenizer = RegexTokenizer(inputCol='Review', outputCol='words', pattern='\\W')
words = tokenizer.transform(df)
words.show(5, truncate=False)

# COMMAND ----------

hashingTF = HashingTF(inputCol='words', outputCol='rawFeatures', numFeatures=20)
featurized = hashingTF.transform(words)
featurized.show(5)

# COMMAND ----------

idf = IDF(inputCol='rawFeatures', outputCol='features')
idf_model = idf.fit(featurized)
rescale = idf_model.transform(featurized)
rescale.select('Liked', 'features').show(5)

# COMMAND ----------

# MAGIC %md # Prototype Model
# MAGIC From what I understand, NLP follows the following 'simple' procedure:
# MAGIC 1) Process text data
# MAGIC 2) Make features numeric (i.e. vectorization)
# MAGIC 3) Train using existing ML models (such as random forest classifier)
# MAGIC 
# MAGIC In my research the 'sparknlp' library provides access to more refined tools for NLP tasks, and so now I will be working with that. My original dataframe from above is still intact, I will be starting there.

# COMMAND ----------

import sparknlp
from sparknlp.base import * # Change once all used modules are known
from sparknlp.annotator import * # Change once all used modules are known
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier, DecisionTreeClassifier, LinearSVC, LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit

# COMMAND ----------

spark = sparknlp.start()

# COMMAND ----------

# MAGIC %md ## Annotation
# MAGIC 
# MAGIC The text needs to be formatted and cleaned as much as possible prior to the vectorization of the data. The steps I wish to implement, to some degree, are as follows. For all of these points, I will modify my approach as new information becomes available to me.
# MAGIC 1) **Tokenization**
# MAGIC     * Splitting of reviews into individual words. This may or may not involve multiple steps, a custom RegEx, or more.
# MAGIC 2) **Spell-checking**
# MAGIC     * All words should be spell-checked prior to any future steps for obvious reasons.
# MAGIC 3) **Stop-words**
# MAGIC     * Removal of unnecessary words that do not contribute significant meaning to the sentence. Removal of these words places more emphasis on the words that matter during training.
# MAGIC 4) **Lemmatization**
# MAGIC     * Words should be reduced to their lemma forms in order to make the text more readable to the computer (i.e. Doing, Does, Did -> Do). In theory, this should strengthen the patterns between certain words and their corresponding meaning as interpreted by the algorithm.

# COMMAND ----------

document = DocumentAssembler()\
                .setInputCol('Review')\
                .setOutputCol('document')\

sentence = SentenceDetector()\
                .setInputCols('document')\
                .setOutputCol('sentence')

tokenizer = Tokenizer()\
                .setInputCols('sentence')\
                .setOutputCol('token')

checker = NorvigSweetingModel()\
                .pretrained()\
                .setInputCols('token')\
                .setOutputCol('checked')

stopwords = StopWordsCleaner()\
                .pretrained('stopwords_en', 'en')\
                .setInputCols('checked')\
                .setOutputCol('cleaned')

lemmatizer = LemmatizerModel()\
                .pretrained()\
                .setInputCols('cleaned')\
                .setOutputCol('lemma')

# COMMAND ----------

pipeline = Pipeline().setStages([document, sentence, tokenizer, checker, stopwords, lemmatizer])
model = pipeline.fit(df)
result = model.transform(df)

# COMMAND ----------

result.select('Review', 'lemma.result').show(truncate=False)

# COMMAND ----------

# MAGIC %md There is an issue with the first line spell checking 'Loved' to 'moved' and eventually lemmatized to 'move'. Perhaps a different tokenizer without punctuation would help. Important negators like 'not' are being flagged by the stopwords filter, and this is a big problem. A review like 'Crust is not good' being turned into 'Crust good' is clearly a flipping of the sentiment of the review. I will need to look into this.
# MAGIC 
# MAGIC I want to create a custom RegEx that will exclude almost all punctuation except for apostrophes and hyphens. As I have zero experience with RegEx, I found a website that allows me to test different RegEx expressions and see how they affect sample text. Using this, I was able to create the RegEx expression r"[a-zA-Z0-9\'\-]" which should include everything I need, and exclude the rest. Since this is being applied after a sentencer, I do not need to worry about single periods.

# COMMAND ----------

tokenizer = RegexTokenizer()\
                .setInputCols('sentence')\
                .setOutputCol('token')\
                .setPattern(r'[^a-zA-Z0-9_\']')

checker = ContextSpellCheckerModel\
                .pretrained()\
                .setInputCols("token")\
                .setOutputCol("checked")

lemmatizer = LemmatizerModel()\
                .pretrained()\
                .setInputCols('checked')\
                .setOutputCol('lemma')

stopwords = StopWordsCleaner()\
                .pretrained('stopwords_iso', 'en')\
                .setInputCols('lemma')\
                .setOutputCol('cleaned')

finisher = Finisher()\
                .setInputCols('lemma')\
                .setOutputCols('result')

# COMMAND ----------

pipeline = Pipeline().setStages([document, sentence, tokenizer, checker, lemmatizer, finisher])
model = pipeline.fit(df)
result = model.transform(df)

# COMMAND ----------

result.show(truncate=False)

# COMMAND ----------

# MAGIC %md A new spell checker, as well as a custom RegEx fixed my issues so far. There are still some errors in the first few rows (i.e. ravoli -> revolt; overpriced -> override), but these are hard to fix and I have to accept some level of error, especially on my first ever NLP project. I still want to explore using a stopword filter, but I'm not quite sure at this point how to ensure I keep contextual descriptors like 'not'. I've tried both pretrained stop-word models available in the spark-nlp library. These are build on the pyspark.ml.features.StopWordsRemover, so that wouldn't work either. My last, very rudimentary option, is to do a custom stop-word list.
# MAGIC 
# MAGIC I'm simply going to look through the first 25 rows and pick out any words I don't believe are necessary. Later on I may use a count vectorizer to find the terms with the most frequency, however this will take a long computation time and for now, I just want to get a prototype working.

# COMMAND ----------

# MAGIC %md ## Vectorization
# MAGIC 
# MAGIC The text now needs to be given a numeric format for usage by the ML classifier. To my knowledge, there are two paradigms in which this can be done; text-frequency based, and pretrained-embeddings based. For this first project, I will be using text-frequency based vectorization, and I will be using a count vectorizer followed by an IDF.

# COMMAND ----------

stop_words = ['the', 'be', 'now', 'get', 'on', 'and', 'so', 'want', 'I', 'it', 'that', 'honesty', 'you', 'they', 'them', 'this', 'by', 'during', 'holiday', 'off', 'back', 'what',
             'say', 'to', 'still', 'end', 'up', 'way', 'little', 'in', 'let', 'alone', 'at', 'all', 'because', 'oh', 'stuff', 'red', 'that\'s', 'a', 'of', 'some']

remover = StopWordsRemover(stopWords = stop_words)\
            .setInputCol('finished')\
            .setOutputCol('cleaned')

finisher = Finisher()\
                .setInputCols('lemma')\
                .setOutputCols('finished')

countVec = CountVectorizer()\
            .setInputCol('finished')\
            .setOutputCol('rawFeatures')

idf = IDF()\
            .setInputCol('rawFeatures')\
            .setOutputCol('features')

pipeline = Pipeline().setStages([document, sentence, tokenizer, checker, lemmatizer, finisher, remover, countVec, idf])
model = pipeline.fit(df)
result = model.transform(df)

result.show()

# COMMAND ----------

# MAGIC %md On second thought, once the model is trained, it would in-fact be useful to have a CountVectorization map rather than a hashmap. The ability to distinguish which buzz-words are associated with positive and negative reviews is quite useful, and it is cumbersome to do this with a hashed count. It only takes around 4-5 minutes to run on this small dataset, so it's not bad at all, actually. I really want to get some predictions going.

# COMMAND ----------

# MAGIC %md ## Model training and evaluation
# MAGIC 
# MAGIC Finally I can train a model on this very rough pipeline. I simply want to get a prototype working so that I can understand all I have until now. Afterwards, I will set up a proper full-length pipeline with several models to try and hyperparameter tuning. For now, I will train on a LogisticRegression classifier (weird name)

# COMMAND ----------

model_data = result.withColumn('label', col('Liked')).select(['features', 'label'])
model_data.show(5)

train, test = model_data.randomSplit([0.9, 0.1], seed=31415)
train.show()

# COMMAND ----------

lr = LogisticRegression()
lrModel = lr.fit(train)

# COMMAND ----------

lr_pred = lrModel.evaluate(test)

# COMMAND ----------

lr_pred.accuracy

# COMMAND ----------

# MAGIC %md # Putting it all together
# MAGIC 
# MAGIC Now I can start looking at training entire pipelined models, applying grid searches on parameters, and testing different models. I will now describe a high-level summary of the tasks to be completed:
# MAGIC 
# MAGIC 1) Pipeline for initial data processing, including the following steps:
# MAGIC     * Document, Sentencing -- Standard
# MAGIC     * Tokenization -- Standard
# MAGIC     * Spell checking -- ContextSpellChecker provided the best results in the exploration phase above
# MAGIC     * Lemmatization -- Standard
# MAGIC     * Stop-word removal -- Need to test several approaches:
# MAGIC         * No stop-words
# MAGIC         * Custom list as used above
# MAGIC         * Pretrained model
# MAGIC     * Finisher (Standard)
# MAGIC     * HashingTF (For speed vs CountVectorizer)
# MAGIC     * IDF Layer (Standard)
# MAGIC     
# MAGIC 2) Model pipelines constructed where previous pipeline ends:
# MAGIC     * One for each model to be tested
# MAGIC     * Use cross validation and parameter grids to identify strongest candidates
# MAGIC     
# MAGIC The model pipelines will be separate from the data processing pipeline due to the fact that the data processing pipeline will be identical regardless of the model used. This will cut down on compute time at the cost of memory. Currently, memory is not an issue.

# COMMAND ----------

# MAGIC %md ## Data Pipelines
# MAGIC 
# MAGIC I will construct three different data pipelines, one for each approach to the stop-word problem.

# COMMAND ----------

!pip install mlflow

!pip install tensorflow

# COMMAND ----------

import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import RegexTokenizer, HashingTF, IDF, StopWordsRemover
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel, TrainValidationSplit
from pyspark.sql.functions import col

import sparknlp
from sparknlp.annotator import * # Change once all used modules are known
from sparknlp.base import * # Change once all used modules are known

# COMMAND ----------

# ---=== Base pipeline with pretrained stopwords ===---

document = DocumentAssembler()\
                .setInputCol('Review')\
                .setOutputCol('document')\

sentence = SentenceDetector()\
                .setInputCols('document')\
                .setOutputCol('sentence')

tokenizer = RegexTokenizer()\
                .setInputCols('sentence')\
                .setOutputCol('token')\
                .setPattern(r'[^a-zA-Z0-9_\']')

checker = ContextSpellCheckerModel\
                .pretrained()\
                .setInputCols("token")\
                .setOutputCol("checked")

lemmatizer = LemmatizerModel()\
                .pretrained()\
                .setInputCols('checked')\
                .setOutputCol('lemma')

stopwords = StopWordsCleaner()\
                .pretrained('stopwords_iso', 'en')\
                .setInputCols('lemma')\
                .setOutputCol('cleaned')

finisher = Finisher()\
                .setInputCols('cleaned')\
                .setOutputCols('finished')

hashingTF = HashingTF()\
                .setInputCol('finished')\
                .setOutputCol('rawFeatures')

idf = IDF()\
                .setInputCol('rawFeatures')\
                .setOutputCol('features')

pipe_sw_pre = Pipeline()\
                .setStages([document, sentence, tokenizer, checker, 
                            lemmatizer, stopwords, finisher, hashingTF, idf])

# ---=== Custom stop words in the pipeline ===---

stop_words = ['the', 'be', 'now', 'get', 'on', 'and', 'so', 'want', 'I', 'it', 'that', 'honesty', 'you', 'they', 'them', 'this', 'by', 'during', 'holiday', 'off', 'back', 'what',
             'say', 'to', 'still', 'end', 'up', 'way', 'little', 'in', 'let', 'alone', 'at', 'all', 'because', 'oh', 'stuff', 'red', 'that\'s', 'a', 'of', 'some']

finisher.setInputCols('lemma')\
        .setOutputCols('finished')

stopwords = StopWordsRemover(stopWords = stop_words)\
                .setInputCol('finished')\
                .setOutputCol('cleaned')

hashingTF.setInputCol('cleaned')\
         .setOutputCol('rawFeatures')
                

pipe_sw_cstm = Pipeline()\
                .setStages([document, sentence, tokenizer, checker, 
                            lemmatizer, finisher, stopwords, hashingTF, idf])

# ---=== No stop words in the pipeline ===---

finisher.setInputCols('lemma')\
        .setOutputCols('finished')

hashingTF.setInputCol('finished')\
         .setOutputCol('rawFeatures')

pipe_sw_none = Pipeline()\
                .setStages([document, sentence, tokenizer, checker, 
                            lemmatizer, finisher, hashingTF, idf])

# COMMAND ----------

# MAGIC %md I wish to use a classification model that allows for explainability of the data. This is a supervised classification problem, and so the possible models I am aware of that meet these criteria are Decision Trees and Logistic Regression. I will start with random-ish values for the parameters and adjust as results come in.

# COMMAND ----------

lr = LogisticRegression()
lr_params = ParamGridBuilder()\
            .addGrid(lr.regParam, [0.1, 0.01])\
            .addGrid(lr.maxIter, [1, 5, 10, 25])\
            .build()

dt = DecisionTreeClassifier()
dt_params = ParamGridBuilder()\
            .addGrid(dt.maxBins, [10,50,100])\
            .addGrid(dt.maxDepth, [10, 50, 100])\
            .build()

# COMMAND ----------

df = sqlContext.sql("Select * FROM nlp_restaurant_reviews")

model_info = {
    'lr' : lr,
    'dt' : dt
}

param_info = {
    'lr' : lr_params,
    'dt' : dt_params
}

pipe_info = {
    'sw_pre' : pipe_sw_pre,
    'sw_cstm' : pipe_sw_cstm,
    'sw_none' : pipe_sw_none
}

df = df.withColumn('label', col('Liked')).select(['label', 'Review'])
train, test = df.randomSplit([0.8, 0.2], seed = 31415)

# , pipe_sw_custom, pipe_sw_none
# , 'dt'

# I will be constructing a JSON-like list containing the results of the calculations.
        
"""
cv = TrainValidationSplit(estimator = pipe_added,
                   estimatorParamMaps = lr_params,
                   evaluator = BinaryClassificationEvaluator(),
                   seed = 31415,
                   parallelism=1
)

cvModel = cv.fit(train)
pred = cvModel.transform(test)
"""

# COMMAND ----------

results_info = {}
evaluator = BinaryClassificationEvaluator()

for pipe_label, pipe in pipe_info.items():
    model_results = {}

    for model_label, model in model_info.items():
        
        pipe_added = Pipeline().setStages([pipe, model])
        pipe_trained = pipe_added.fit(train)
        pipe_data = pipe_trained.transform(train)

        areaUnderROC = evaluator.evaluate(pipe_data)
        print(f'{pipe_label} - {model_label}: {areaUnderROC}')
        
        model_results[model_label] = areaUnderROC
    results_info[pipe_label] = model_results

# COMMAND ----------


