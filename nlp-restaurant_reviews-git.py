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
lr.fit(train)

# COMMAND ----------


