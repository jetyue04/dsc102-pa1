from dask.distributed import as_completed

def get_supercat(cat_str):
    try:
        cat_list = ast.literal_eval(cat_str)
        if isinstance(cat_list, list) and cat_list and isinstance(cat_list[0], list) and cat_list[0]:
            return cat_list[0][0]
    except:
        return None
    return None

def check_dangling(partition, products_subset):
    merged = partition[['asin']].merge(products_subset, on='asin', how='left', indicator=True)
    return int((merged['_merge'] == 'left_only').any().compute())

def safe_extract(related_str):
    try:
        d = ast.literal_eval(related_str)
        ids = []
        for k in ['also_bought', 'also_viewed', 'bought_together', 'buy_after_viewing']:
            val = d.get(k)
            if isinstance(val, list):
                ids.extend(val)
        return ids
    except:
        return []

def check_related_partition(partition, valid_asins_pd):
    related_series = partition['related'].dropna().apply(safe_extract)
    exploded = related_series.explode().dropna()
    exploded_df = exploded.to_frame(name='asin')
    merged = exploded_df.merge(valid_asins_pd, on='asin', how='left', indicator=True)
    return int((merged['_merge'] == 'left_only').any().compute())

reviews = dd.read_csv('user_reviews.csv', dtype={'asin': 'object'})
products = dd.read_csv('products.csv', dtype={'asin': 'object'})

reviews = reviews.persist()
products = products.persist()

### Question 1
q1 = reviews.isna().sum() / len(reviews) * 100

### Question 2
q2 = products.isna().sum() / len(products) * 100
merged = dd.merge(
    reviews[['asin', 'overall']],
    products[['asin', 'price']],
    on='asin'
    how='inner'
).dropna(subset=['overall', 'price'])

### Question 3
q3 = merged[['overall', 'price']].corr('pearson',numeric_only=True)
price = products['price'].dropna()

### Question 4
q4 = (price.mean(),
      price.std(),
      price.min(),
      price.max(),
      price.quantile(0.5)
     )

### Question 5
q5 = products['categories'].dropna().map(get_supercat, meta=('supercat', 'object')).value_counts()

ans1_5 = dd.compute(q1,q2,q3,q4,q5)
ans1_5
product_asins_df_future = client.scatter(products[['asin']], broadcast=True)

delayed_parts = reviews.to_delayed()
futures = [client.submit(check_dangling, part, product_asins_df_future) for part in delayed_parts]

### Question 6
q6 = 0
for fut in as_completed(futures):
    if fut.result():
        q6 = 1
        break
client.cancel(futures)

### Question 7
delayed_parts = products[['related']].dropna().to_delayed()
futures = [client.submit(check_related_partition, part, product_asins_df_future) for part in delayed_parts]

q7 = 0
for fut in as_completed(futures):
    if fut.result():
        q7 = 1
        break
client.cancel(futures)
