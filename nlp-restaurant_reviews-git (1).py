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

countTokens = udf(lambda p: len(p), IntegerType()) # udf applies the lambda function to each row

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

# MAGIC %md # Implementation
# MAGIC From what I understand, NLP follows the following 'simple' procedure:
# MAGIC 1) Process text data
# MAGIC 2) Make features numeric (i.e. vectorization)
# MAGIC 3) Train using existing ML models (such as random forest classifier)
# MAGIC 
# MAGIC In my research, the above work was nice, but it seems like the library 'sparknlp' provides access to more refined tools for NLP tasks, and so now I will be working with that. My original dataframe from above is still intact, I will be starting there.

# COMMAND ----------

import sparknlp
from sparknlp.base import * # Change once all used modules are known
from sparknlp.annotator import * # Change once all used modules are known
from pyspark.ml import Pipeline

# COMMAND ----------

spark = sparknlp.start()

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

stopwords = StopWordsCleaner()\
                .pretrained('stopwords_en', 'en')\
                .setInputCols('token')\
                .setOutputCol('trimmed')

checker = NorvigSweetingModel()\
                .pretrained()\
                .setInputCols('trimmed')\
                .setOutputCol('checked')

lemmatizer = LemmatizerModel().pretrained()\
                .setInputCols('checked')\
                .setOutputCol('lemma')

# COMMAND ----------

checker = NorvigSweetingModel()\
                .pretrained()\
                .setInputCols('trimmed')\
                .setOutputCol('checked')

# COMMAND ----------

pipeline = Pipeline().setStages([document, sentence, tokenizer, stopwords, checker, lemmatizer])
model = pipeline.fit(df)
result = model.transform(df)

# COMMAND ----------

result.select('Review', 'lemma.result').show(truncate=False)

# COMMAND ----------

# MAGIC %md In this particular case it looks like the stop words are actually not helping. It is removing key words like 'not' which can instantly turn the connotation of the review from bad to good, providing a false signal. There is also an issue with the first line spell checking 'Loved' to 'moved' and eventually lemmatized to 'move'. Perhaps a different tokenizer without punctuation would help.

# COMMAND ----------

tokenizer = RegexTokenizer()\
                .setInputCols('sentence')\
                .setOutputCol('token')\
                .setPattern(r'[^a-zA-z0-9_\']')

# COMMAND ----------

checker = ContextSpellCheckerModel()\
                .pretrained()\
                .setInputCols('token')\
                .setOutputCol('checked')

# COMMAND ----------

pipeline = Pipeline().setStages([document, sentence, tokenizer, checker, lemmatizer])
model = pipeline.fit(df)
result = model.transform(df)

# COMMAND ----------

result.select('Review', 'checked.result').show(truncate=False)

# COMMAND ----------

# MAGIC %md A new spell checker, as well as a custom RegEx fixed my issues so far. There are still some errors in the first few rows (i.e. ravoli -> revolt; overpriced -> override), but these are hard to fix and I have to accept some level of error. Now to see how the Lemmatizer has gone so far. I still want to explore using a stopword filter, but I'm not quite sure at this point how to ensure I keep contextual descriptors like 'not'.

# COMMAND ----------

result.select('Review', 'lemma.result').show(truncate=False)

# COMMAND ----------


