#!/usr/bin/env python
# coding: utf-8

# In[11]:


#load and preprocess the data
import pandas as pd

# Replace file paths with the paths to the downloaded files
orders_df = pd.read_csv("C:/Users/lydi_/OneDrive/Documents/DTU master , lectures and exercises/Computational Tools for Data Science/olist_orders_dataset.csv")
order_items_df = pd.read_csv("C:/Users/lydi_/OneDrive/Documents/DTU master , lectures and exercises/Computational Tools for Data Science/olist_order_items_dataset.csv")
products_df = pd.read_csv("C:/Users/lydi_/OneDrive/Documents/DTU master , lectures and exercises/Computational Tools for Data Science/olist_products_dataset.csv")
reviews_df = pd.read_csv("C:/Users/lydi_/OneDrive/Documents/DTU master , lectures and exercises/Computational Tools for Data Science/olist_order_reviews_dataset.csv")


# In[12]:


#checking for missing values 
print(orders_df.isnull().sum())
print(order_items_df.isnull().sum())
print(products_df.isnull().sum())
print(reviews_df.isnull().sum())
# Repeat for other datasets


# In[13]:


# Impute categorical data with the mode (most frequent category)
products_df['product_category_name'].fillna(products_df['product_category_name'].mode()[0], inplace=True)

# Impute numerical data with the mean
for col in ['product_name_lenght', 'product_description_lenght', 'product_photos_qty']:
    products_df[col].fillna(products_df[col].mean(), inplace=True)

# For very few missing values in product dimensions and weight, use mean
for col in ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']:
    products_df[col].fillna(products_df[col].mean(), inplace=True)

# For review comments, replace missing values with 'No Comment'
reviews_df['review_comment_title'].fillna('No Comment', inplace=True)
reviews_df['review_comment_message'].fillna('No Comment', inplace=True)


# In[14]:


#after handling the missing values we recheck the datasets to ensure that all missing values have been appropriately addressed
print("Missing values in Product Dataset:")
print(products_df.isnull().sum())

print("\nMissing values in Review Dataset:")
print(reviews_df.isnull().sum())


# In[16]:


#merging the datasets
# Example of merging orders and order items
full_order_df = pd.merge(orders_df, order_items_df, on='order_id', how='left')

# Example of merging the full order data with product information
full_order_product_df = pd.merge(full_order_df, products_df, on='product_id', how='left')



# In[18]:


#creating new features 
#based on the merged data, we want to create new features that can help in our analysis or model building

full_order_product_df['total_value'] = full_order_product_df['price'] + full_order_product_df['freight_value']

# Example: Convert timestamp to datetime and extract useful parts
full_order_product_df['order_purchase_timestamp'] = pd.to_datetime(full_order_product_df['order_purchase_timestamp'])
full_order_product_df['purchase_weekday'] = full_order_product_df['order_purchase_timestamp'].dt.day_name()
full_order_product_df['purchase_hour'] = full_order_product_df['order_purchase_timestamp'].dt.hour


# In[20]:


#after we create the new features we do the final check for missing values and data types
# Check for missing values in the new dataframe
print(full_order_product_df.isnull().sum())

# Check data types
print(full_order_product_df.dtypes)

# Save the processed dataframe to a new CSV for easier access in the future
full_order_product_df.to_csv('processed_data.csv', index=False)
    


# In[21]:


#handling missing values and data types


# In[23]:


# Handling missing values
# Dropping rows where order items are missing
full_order_product_df.dropna(subset=['order_item_id'], inplace=True)

# Filling missing dates with placeholder or imputation
full_order_product_df['order_approved_at'].fillna(method='ffill', inplace=True)  # Example: forward fill

# Converting date columns to datetime
full_order_product_df['order_approved_at'] = pd.to_datetime(full_order_product_df['order_approved_at'])

# Save the processed dataframe
full_order_product_df.to_csv('processed_data.csv', index=False)


# In[29]:


#create a simple recommendatuin system

# Merging datasets to get the user, product, and category information together
merged_df = pd.merge(orders_df, order_items_df, on='order_id')
merged_df = pd.merge(merged_df, products_df, on='product_id')

# Assuming each customer has a unique customer_id
def recommend_products(customer_id, num_recommendations=5):
    # Find products previously ordered by the customer
    ordered_products = merged_df[merged_df['customer_id'] == customer_id]['product_category_name']
    
    # Recommend other products in the same categories
    recommendations = merged_df[merged_df['product_category_name'].isin(ordered_products)]
    return recommendations['product_id'].unique()[:num_recommendations]

# Example usage
#customer_id = 'some_customer_id'  # replace with an actual customer ID from the dataset
#recommended_products = recommend_products(customer_id)
#print(recommended_products)


# 
# ### Collaborative Filtering
# 
# Collaborative filtering is a method used in recommendation systems to predict the preferences of one user based on the preferences of other users. The basic assumption is that if a person A has the same opinion as a person B on an issue, A is more likely to have B's opinion on a different issue than that of a random person. There are two main types:
# 
# 1. **User-Based Collaborative Filtering**: This approach recommends items by finding similar users. For example, if user X and user Y both liked items A and B, and user X likes item C, the system might recommend item C to user Y. The similarity between users is usually calculated using methods like cosine similarity or Pearson correlation.
# 
# 2. **Item-Based Collaborative Filtering**: Instead of finding similar users, this approach finds similar items based on user ratings. For instance, if item A and item B are both highly rated by many users who rate both, and a user likes item A, the system might recommend item B to that user. Again, similarity can be measured by cosine similarity or other metrics.
# 
# ### Machine Learning Models in Recommendation Systems
# 
# Machine learning models can be used in recommendation systems to predict user preferences and recommend items. These models can either be used independently or in conjunction with collaborative filtering. Common approaches include:
# 
# 1. **Classification Models**: These models can classify whether a user would like or dislike an item. Algorithms like logistic regression, decision trees, or support vector machines can be used.
# 
# 2. **Regression Models**: If the rating system is numerical, regression models can predict the rating a user might give to an item. Algorithms like linear regression or random forests can be used.
# 
# 3. **Matrix Factorization Techniques**: These are more advanced techniques used in recommendation systems, like Singular Value Decomposition (SVD). They work by decomposing the user-item interaction matrix into the product of two lower dimensionality rectangular matrices.
# 
# 4. **Neural Networks and Deep Learning**: Deep learning models can be used for more complex recommendation systems. They are particularly useful for handling large-scale and sparse datasets and can capture complex non-linear relationships between users and items.
# 
# 5. **Hybrid Models**: These models combine collaborative filtering with other machine learning techniques. For example, a hybrid model might use collaborative filtering to find a user's preferences and then a classification model to predict whether the user will like a new item.
# 
# In practice, the choice of model depends on the specific requirements of the recommendation system, such as the size and nature of the dataset, the type of recommendations required (binary like/dislike, ratings, etc.), and the computational resources available. Each model has its strengths and weaknesses, and often a combination of these approaches yields the best results.

# #Step 3: Improvements and Machine Learning
# #Collaborative Filtering Techniques:
# 
# #Collaborative filtering can be implemented in two main ways: user-based and item-based.
# #User-Based: Recommends items by finding similar users. This is often effective but can be computationally expensive.
# #Item-Based: Recommends items similar to those the user has liked before. It's generally faster and more stable than user-based.

# In[42]:


# Step 1: Merge Review Scores
merged_df = pd.merge(full_order_product_df, reviews_df[['order_id', 'review_score']], on='order_id', how='left')

# Step 2: Create 'Rating' Column
# Handling missing values - fill with average score or a predefined score
# You can also choose to fill with median or any other statistical measure
default_rating = merged_df['review_score'].mean()
merged_df['rating'] = merged_df['review_score'].fillna(default_rating)

# Now, merged_df has a 'rating' column, which you can use for the collaborative filtering


# In[47]:


# Step 1: Examine the Dataframes
print("Full Order Product DataFrame:")
print(full_order_product_df.head())

print("\nReviews DataFrame:")
print(reviews_df.head())

# Step 2: Verify the 'order_id' column in both dataframes
print("\nColumn 'order_id' in full_order_product_df:", 'order_id' in full_order_product_df.columns)
print("Column 'order_id' in reviews_df:", 'order_id' in reviews_df.columns)

# Step 3: Re-attempt the Merge
# Assuming the 'order_id' column exists in both dataframes and 'review_score' is in reviews_df
merged_df = pd.merge(full_order_product_df, reviews_df[['order_id', 'review_score']], on='order_id', how='left')

# Step 4: Create the 'Rating' Column
default_rating = merged_df['review_score'].mean()
merged_df['rating'] = merged_df['review_score'].fillna(default_rating)

# Step 5: Check if 'rating' column is added
print("\nRating column added:", 'rating' in merged_df.columns)


# In[50]:


#1. Reduce the Data Size
#You might want to filter the data to include only the most popular products or the most active customers. Here's a basic way to do it:

# Example: Selecting top N products based on the number of orders
top_products = merged_df['product_id'].value_counts().head(1000).index
filtered_df = merged_df[merged_df['product_id'].isin(top_products)]

#  create the pivot table with this filtered dataframe
pivot_table = filtered_df.pivot_table(index='customer_id', columns='product_id', values='rating').fillna(0)

#2. Use Sparse Matrix

from scipy.sparse import csr_matrix

# Create a sparse pivot table
pivot_table_sparse = csr_matrix(pivot_table.fillna(0))

#3. Check Data Types

merged_df['customer_id'] = merged_df['customer_id'].astype(str)
merged_df['product_id'] = merged_df['product_id'].astype(str)


# In[55]:


# Create the pivot table
pivot_table = merged_df.pivot_table(index='customer_id', columns='product_id', values='rating').fillna(0)

# Compute the cosine similarity
item_similarity = cosine_similarity(pivot_table.T)
item_similarity_df = pd.DataFrame(item_similarity, index=pivot_table.columns, columns=pivot_table.columns)

# Function to make recommendations
def recommend_products(customer_id, n_items=5):
    if customer_id not in pivot_table.index:
        raise ValueError("Customer ID not found in the data.")
    
    customer_ratings = pivot_table.loc[customer_id]
    similar_scores = item_similarity_df[customer_ratings.index].dot(customer_ratings.values)
    similar_scores = similar_scores.sort_values(ascending=False)
    return similar_scores.index[:n_items]



# In[57]:


from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import numpy as np

# Create a sparse matrix for customer-product interactions

customer_ids = pd.factorize(merged_df['customer_id'])[0]
product_ids = pd.factorize(merged_df['product_id'])[0]
ratings = merged_df['rating'].values

sparse_matrix = csr_matrix((ratings, (customer_ids, product_ids)), shape=(len(np.unique(customer_ids)), len(np.unique(product_ids))))

# Compute cosine similarity between items
item_similarity = cosine_similarity(sparse_matrix.T, dense_output=False)


# In[60]:


product_id_mapping = pd.Series(index=np.unique(product_ids), data=merged_df['product_id'].unique())

def recommend_products(product_id, top_n=5):
    # Find the internal index of the product
    product_idx = product_id_mapping[product_id_mapping == product_id].index[0]
    
    # Get similarity values
    similarity_values = item_similarity[product_idx].toarray().flatten()
    
    # Get indices of top similar products
    similar_product_indices = similarity_values.argsort()[::-1][1:top_n+1]  # Exclude the product itself
    
    # Convert these indices back to product IDs
    similar_products = product_id_mapping.iloc[similar_product_indices].values

    return similar_products


# Now we will find an example product ID from our dataset :
# 1.First we will take a look at a few entries in our merged_df dataframe to identify product IDs
# 2.We will choose a product ID for testing
# 

# In[62]:


print(merged_df['product_id'].head())


# In[66]:


print(recommend_products('87285b34884572647811a353c7ac498a'))

