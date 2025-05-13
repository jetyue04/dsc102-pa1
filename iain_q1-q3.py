### read in the 'user_reviews.csv' and 'products.csv' files, perform your calculations and place the answers in variables ans1 - ans7.

reviews = dd.read_csv('user_reviews.csv', blocksize="100MB", dtype={'asin': 'object'})
products = dd.read_csv('products.csv', blocksize="100MB", dtype={'asin': 'object'})


# question 1
q1 = (reviews.isna().sum() / len(reviews)) * 100
ans1 = q1.compute()


# question 2
q2 = (products.isna().sum() / len(products)) * 100
ans2 = q2.compute()


# question 3
merged = reviews[['asin', 'overall']].merge(
    products[['asin', 'price']], 
    on='asin', 
    how='inner')

merged = merged.dropna(subset=['price', 'overall'])
merged['price'] = merged['price'].astype(float)
merged['overall'] = merged['overall'].astype(float)

q3 = merged_clean[['price', 'overall']].corr()
ans3 = q3.compute().loc['price', 'overall']

# question 4
ans4 = None

# question 5
ans5 = None

# question 6
ans6 = None

# question 7
ans7 = None