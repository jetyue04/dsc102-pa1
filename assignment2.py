import os
import pyspark.sql.functions as F
import pyspark.sql.types as T
from utilities import SEED
# import any other dependencies you want, but make sure only to use the ones
# availiable on AWS EMR

# ---------------- choose input format, dataframe or rdd ----------------------
INPUT_FORMAT = 'dataframe'  # change to 'rdd' if you wish to use rdd inputs
# -----------------------------------------------------------------------------
if INPUT_FORMAT == 'dataframe':
    import pyspark.ml as M
    import pyspark.sql.functions as F
    import pyspark.sql.types as T
    from pyspark.ml.regression import DecisionTreeRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
if INPUT_FORMAT == 'koalas':
    import databricks.koalas as ks
elif INPUT_FORMAT == 'rdd':
    import pyspark.mllib as M
    from pyspark.mllib.feature import Word2Vec
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.linalg.distributed import RowMatrix
    from pyspark.mllib.tree import DecisionTree
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.linalg import DenseVector
    from pyspark.mllib.evaluation import RegressionMetrics
    
#### OWN IMPORTS ####
from pyspark.sql.types import StringType, IntegerType, MapType
from pyspark.ml.linalg import DenseVector
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.feature import Word2Vec


# ---------- Begin definition of helper functions, if you need any ------------

# def task_1_helper():
#   pass
def preprocess_title(title):
        return title.lower().split() if title else []
# -----------------------------------------------------------------------------


def task_1(data_io, review_data, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    overall_column = 'overall'
    # Outputs:
    mean_rating_column = 'meanRating'
    count_rating_column = 'countRating'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    overall_column = 'overall'
    # Outputs:
    mean_rating_column = 'meanRating'
    count_rating_column = 'countRating'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    #     review_data.show(5)
#     product_data.show(5)

    # Extract useful columns
    review_df = review_data.select(asin_column, overall_column)
    product_df = product_data.select(asin_column)
#     review_df.show(5)
#     product_df.show(5)
    
    # Calculate average and count for each product
    agg_review = review_df.groupby(asin_column).agg(
        F.avg(overall_column).alias(mean_rating_column),
        F.count(overall_column).alias(count_rating_column)
    )
    
    # Left join product_df with agg_review
    merged_df = product_df.join(agg_review, on=asin_column, how='left')


    #Cache dataset
    merged_df.cache()
    
    #Compute
    count_total = merged_df.count()
    
    mean_variance_stats = merged_df.select(
        F.mean(mean_rating_column).alias('mean_meanRating'),
        F.variance(mean_rating_column).alias('variance_meanRating'),
        F.mean(count_rating_column).alias('mean_countRating'),
        F.variance(count_rating_column).alias('variance_countRating')
    ).first()
    
    numNulls_meanRating = merged_df.filter(F.col(mean_rating_column).isNull()).count()
    numNulls_countRating = merged_df.filter(F.col(count_rating_column).isNull()).count()

    merged_df.unpersist()

    #Extract answers
    mean_meanRating = mean_variance_stats['mean_meanRating']
    variance_meanRating = mean_variance_stats['variance_meanRating']
    mean_countRating = mean_variance_stats['mean_countRating']
    variance_countRating = mean_variance_stats['variance_countRating']
    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    # Calculate the values programmaticly. Do not change the keys and do not
    # hard-code values in the dict. Your submission will be evaluated with
    # different inputs.
    # Modify the values of the following dictionary accordingly.
    res = {
        'count_total': count_total,
        'mean_meanRating': mean_meanRating,
        'variance_meanRating': variance_meanRating,
        'numNulls_meanRating': numNulls_meanRating,
        'mean_countRating': mean_countRating,
        'variance_countRating': variance_countRating,
        'numNulls_countRating': numNulls_countRating
    }
    # Modify res:
    
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_1')
    return res
    # -------------------------------------------------------------------------


def task_2(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    salesRank_column = 'salesRank'
    categories_column = 'categories'
    asin_column = 'asin'
    # Outputs:
    category_column = 'category'
    bestSalesCategory_column = 'bestSalesCategory'
    bestSalesRank_column = 'bestSalesRank'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
## Extract salesrank key pair
    product_df = product_data.withColumn(
        bestSalesCategory_column,
        F.when(
            (F.col(salesRank_column).isNull()) | (F.size(F.map_keys(F.col(salesRank_column))) == 0), 
            None
        ).otherwise(F.map_keys(F.col(salesRank_column))[0])
    ).withColumn(
        bestSalesRank_column,
        F.when(
            (F.col(salesRank_column).isNull()) | (F.size(F.map_values(F.col(salesRank_column))) == 0), 
            None
        ).otherwise(F.map_values(F.col(salesRank_column))[0])
    )
    product_df.show()
    ## Extract category column
    product_df = product_df.withColumn(
        category_column,
        F.when(
            (F.col(categories_column)[0][0] == "") | (F.col(categories_column)[0][0].isNull()) | (F.size(F.col(categories_column)) == 0),
            None
        ).otherwise(F.col(categories_column)[0][0])
    )
    
#     product_df.show()
    
    
    
    ## Compute
    count_total = product_df.count()
    mean_bestSalesRank = product_df.select(F.mean(F.col(bestSalesRank_column))).first()[0]
    variance_bestSalesRank = product_df.select(F.variance(F.col(bestSalesRank_column))).first()[0]
    numNulls_category = product_df.filter(F.col(category_column).isNull()).count()
    countDistinct_category = product_df.filter(F.col(category_column).isNotNull()).select(category_column).distinct().count()
    numNulls_bestSalesCategory = product_df.filter(F.col(bestSalesCategory_column).isNull()).count()
    countDistinct_bestSalesCategory = product_df.filter(F.col(bestSalesCategory_column).isNotNull()).select(bestSalesCategory_column).distinct().count()

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': count_total,
        'mean_bestSalesRank': float(mean_bestSalesRank),
        'variance_bestSalesRank': float(variance_bestSalesRank),
        'numNulls_category': numNulls_category,
        'countDistinct_category': countDistinct_category,
        'numNulls_bestSalesCategory': numNulls_bestSalesCategory,
        'countDistinct_bestSalesCategory': countDistinct_bestSalesCategory
    }
    # Modify res:




    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_2')
    return res
    # -------------------------------------------------------------------------


def task_3(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    price_column = 'price'
    attribute = 'also_viewed'
    related_column = 'related'
    # Outputs:
    meanPriceAlsoViewed_column = 'meanPriceAlsoViewed'
    countAlsoViewed_column = 'countAlsoViewed'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    exploded = product_data.select(
        F.col(asin_column).alias('main_asin'),
        F.explode_outer(F.col(f"{related_column}.{attribute}")).alias('also_viewed_asin')
    ).join(
        product_data.select(
            F.col(asin_column).alias('also_viewed_asin'),
            F.col(price_column)
        ),
        on="also_viewed_asin",
        how="left"
    )
    
#     exploded.show(5)
    
    #Calculate mean of each row ignoring null
    mean_df = exploded.filter(F.col(price_column).isNotNull()).groupBy("main_asin").agg(
        F.mean(price_column).alias(meanPriceAlsoViewed_column)
    )
#     mean_df.show(5)
    
    #Count number related product for each product
    count_df = exploded.groupBy("main_asin").agg(
        F.count("also_viewed_asin").alias(countAlsoViewed_column)
    )
#     count_df.show(5)
    
    
    result = product_data.select(asin_column, related_column) \
        .join(mean_df, product_data[asin_column] == mean_df["main_asin"], how="left") \
        .drop(mean_df["main_asin"]) \
        .join(count_df, product_data[asin_column] == count_df["main_asin"], how="left") \
        .drop(count_df["main_asin"])
    
    result = result.withColumn(
        meanPriceAlsoViewed_column,
        F.when(
            F.col(f"{related_column}.{attribute}").isNull() | (F.size(F.col(f"{related_column}.{attribute}")) == 0),
            None
        ).otherwise(F.col(meanPriceAlsoViewed_column))
    ).withColumn(
        countAlsoViewed_column,
        F.when(
            F.col(f"{related_column}.{attribute}").isNull() | (F.size(F.col(f"{related_column}.{attribute}")) == 0),
            None
        ).otherwise(F.col(countAlsoViewed_column))
    )

    
    
    
#     result.show(5)
    
    count_total = result.count()
    mean_meanPriceAlsoViewed = result.select(F.mean(F.col(meanPriceAlsoViewed_column))).first()[0]
    variance_meanPriceAlsoViewed = result.select(F.variance(F.col(meanPriceAlsoViewed_column))).first()[0]
    numNulls_meanPriceAlsoViewed = result.filter(F.col(meanPriceAlsoViewed_column).isNull()).count()
    mean_countAlsoViewed = result.select(F.mean(F.col(countAlsoViewed_column))).first()[0]
    variance_countAlsoViewed = result.select(F.variance(F.col(countAlsoViewed_column))).first()[0]
    numNulls_countAlsoViewed = result.filter(F.col(countAlsoViewed_column).isNull()).count()




    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': count_total,
        'mean_meanPriceAlsoViewed': float(mean_meanPriceAlsoViewed),
        'variance_meanPriceAlsoViewed': float(variance_meanPriceAlsoViewed),
        'numNulls_meanPriceAlsoViewed': numNulls_meanPriceAlsoViewed,
        'mean_countAlsoViewed': float(mean_countAlsoViewed),
        'variance_countAlsoViewed': float(variance_countAlsoViewed),
        'numNulls_countAlsoViewed': numNulls_countAlsoViewed
    }
    # Modify res:




    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_3')
    return res
    # -------------------------------------------------------------------------


def task_4(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    price_column = 'price'
    title_column = 'title'
    # Outputs:
    meanImputedPrice_column = 'meanImputedPrice'
    medianImputedPrice_column = 'medianImputedPrice'
    unknownImputedTitle_column = 'unknownImputedTitle'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    product_data = product_data.withColumn(price_column, F.col(price_column).cast('float'))
    mean = product_data.select(F.mean(F.col(price_column))).first()[0]
    median = product_data.approxQuantile(price_column, [0.5], 0.01)[0]
    
    
    # adds in median inputed col:
    product_data = product_data.withColumn(medianImputedPrice_column, F.when(F.col(price_column).isNull(), median).otherwise(F.col(price_column)))

    
    # adds in mean inputed col:
    product_data = product_data.withColumn(meanImputedPrice_column, F.when(F.col(price_column).isNull(), mean).otherwise(F.col(price_column)))
    
    # adds in unknown col filled 
    product_data = product_data.withColumn(unknownImputedTitle_column,F.when(F.col(title_column).isNull() | (F.col(title_column) == ""), "unknown").otherwise(F.col(title_column)))
    
    # Calculate statistics for meanImputedPrice
    meanImputedPrice_col = F.col(meanImputedPrice_column)
    
    mean_meanImputedPrice = product_data.select(F.mean(meanImputedPrice_col)).first()[0]
    variance_meanImputedPrice = product_data.select(F.variance(meanImputedPrice_col)).first()[0]
    numNulls_meanImputedPrice = product_data.filter(meanImputedPrice_col.isNull()).count()

    # Calculate statistics for medianImputedPrice
    medianImputedPrice_col = F.col(medianImputedPrice_column)
    
    mean_medianImputedPrice = product_data.select(F.mean(medianImputedPrice_col)).first()[0]
    variance_medianImputedPrice = product_data.select(F.variance(medianImputedPrice_col)).first()[0]
    numNulls_medianImputedPrice = product_data.filter(medianImputedPrice_col.isNull()).count()

    # Calculate number of 'unknown' values in unknownImputedTitle
    numUnknowns_unknownImputedTitle = product_data.filter(F.col(unknownImputedTitle_column) == "unknown").count()

    # Count total number of rows
    count_total = product_data.count()




    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': count_total,
        'mean_meanImputedPrice': mean_meanImputedPrice,
        'variance_meanImputedPrice': variance_meanImputedPrice,
        'numNulls_meanImputedPrice': numNulls_meanImputedPrice,
        'mean_medianImputedPrice': mean_medianImputedPrice,
        'variance_medianImputedPrice': variance_medianImputedPrice,
        'numNulls_medianImputedPrice': numNulls_medianImputedPrice,
        'numUnknowns_unknownImputedTitle': numUnknowns_unknownImputedTitle
    }
    # Modify res:




    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_4')
    return res
    # -------------------------------------------------------------------------


def task_5(data_io, product_processed_data, word_0, word_1, word_2):
    # -----------------------------Column names--------------------------------
    # Inputs:
    title_column = 'title'
    # Outputs:
    titleArray_column = 'titleArray'
    titleVector_column = 'titleVector'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    preprocess_udf = F.udf(preprocess_title, T.ArrayType(T.StringType()))
    product_prepared = product_processed_data.withColumn(titleArray_column, preprocess_udf(F.col(title_column)))

    
    word2vec = Word2Vec(
        inputCol=titleArray_column,
        outputCol="dummy",  # Output column not used, just required by API
        vectorSize=16,
        minCount=100,
        numPartitions=4,
        seed=SEED
    )
    model = word2vec.fit(product_prepared)

  
    synonyms_word_0 = [(row.word, row.similarity) for row in model.findSynonyms(word_0, 10).collect()]
    synonyms_word_1 = [(row.word, row.similarity) for row in model.findSynonyms(word_1, 10).collect()]
    synonyms_word_2 = [(row.word, row.similarity) for row in model.findSynonyms(word_2, 10).collect()]

    
    count_total = product_prepared.count()
    size_vocabulary = model.getVectors().count()



    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': count_total,
        'size_vocabulary': size_vocabulary,
        'word_0_synonyms': synonyms_word_0,
        'word_1_synonyms': synonyms_word_1,
        'word_2_synonyms': synonyms_word_2
    }
    # Modify res:



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_5')
    return res
    # -------------------------------------------------------------------------


def task_6(data_io, product_processed_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    category_column = 'category'
    # Outputs:
    categoryIndex_column = 'categoryIndex'
    categoryOneHot_column = 'categoryOneHot'
    categoryPCA_column = 'categoryPCA'
    # -------------------------------------------------------------------------    

    # ---------------------- Your implementation begins------------------------
    product_processed_data.show(5)

    ### One hot encode features
    indexer = M.feature.StringIndexer(
        inputCol=category_column, 
        outputCol=categoryIndex_column
    )
    product_indexed = indexer.fit(product_processed_data).transform(product_processed_data)
    
    product_indexed.show(5)

    encoder = M.feature.OneHotEncoder(
        inputCol=categoryIndex_column, 
        outputCol=categoryOneHot_column, 
        dropLast=False
    )
    product_encoded = encoder.fit(product_indexed).transform(product_indexed)
    
    product_encoded.show(5)
    ### Apply PCA
    pca = M.feature.PCA(k=15, 
                        inputCol=categoryOneHot_column, 
                        outputCol=categoryPCA_column
    )
    pca_data = pca.fit(product_encoded).transform(product_encoded)
    pca_data.show(5)
    
    first_vector = pca_data.select(categoryPCA_column).first()[0]
    print(type(first_vector))
    print(first_vector)
    ### Compute
    count_total = pca_data.count()
    
    #Mean vectors
    onehot_mean = pca_data.select(M.stat.Summarizer.mean(F.col(categoryOneHot_column))).first()[0]
    meanVector_categoryOneHot = DenseVector(onehot_mean.toArray()).tolist()
    
    pca_mean = pca_data.select(M.stat.Summarizer.mean(F.col(categoryPCA_column))).first()[0]
    meanVector_categoryPCA = DenseVector(pca_mean.toArray()).tolist()
    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': count_total,
        'meanVector_categoryOneHot': meanVector_categoryOneHot,
        'meanVector_categoryPCA': meanVector_categoryPCA
    }
    # Modify res:




    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_6')
    return res
    # -------------------------------------------------------------------------
    
    
def task_7(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    train_data.show(5)
    test_data.show(5)
    dt = DecisionTreeRegressor(featuresCol='features', labelCol='overall', maxDepth=5)

    #Train
    dt_model = dt.fit(train_data)

    #predict
    predictions = dt_model.transform(test_data)

    #Eval
    evaluator = RegressionEvaluator(labelCol="overall", predictionCol="prediction", metricName="rmse")
    test_rmse = evaluator.evaluate(predictions)
    
    
    
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': test_rmse
    }
    # Modify res:


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_7')
    return res
    # -------------------------------------------------------------------------
    
    
def task_8(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    train_data.show(5)
    test_data.show(5)
    
    train_df, valid_df = train_data.randomSplit([0.75, 0.25])

    max_depths = [5, 7, 9, 12]

    valid_rmse_dict = {}

    evaluator = RegressionEvaluator(metricName="rmse", labelCol="overall", predictionCol="prediction")

    for depth in max_depths:
        # train
        dt = DecisionTreeRegressor(maxDepth=depth, labelCol="overall", featuresCol="features")
        model = dt.fit(train_df)
        
        # predict + eval on validation
        valid_predictions = model.transform(valid_df)
        
        # eval
        evaluator = RegressionEvaluator(labelCol="overall", predictionCol="prediction", metricName="rmse")
        valid_rmse = evaluator.evaluate(valid_predictions)
        valid_rmse_dict[depth] = valid_rmse

    # find best depth
    best_depth = min(valid_rmse_dict, key=valid_rmse_dict.get)
    best_rmse = valid_rmse_dict[best_depth]
    best_model = DecisionTreeRegressor(maxDepth=best_depth, labelCol="overall", featuresCol="features").fit(train_data)

    #eval on test
    test_predictions = best_model.transform(test_data)
    test_rmse = evaluator.evaluate(test_predictions)
    
    
    
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': test_rmse,
        'valid_rmse_depth_5': valid_rmse_dict.get(5),
        'valid_rmse_depth_7': valid_rmse_dict.get(7),
        'valid_rmse_depth_9': valid_rmse_dict.get(9),
        'valid_rmse_depth_12': valid_rmse_dict.get(12)
    }
    # Modify res:


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_8')
    return res
    # -------------------------------------------------------------------------

