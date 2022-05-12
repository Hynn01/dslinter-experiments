#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Now that you can select raw data, you're ready to learn how to group your data and count things within those groups. This can help you answer questions like: 
# 
# * How many of each kind of fruit has our store sold?
# * How many species of animal has the vet office treated?
# 
# To do this, you'll learn about three new techniques: **GROUP BY**, **HAVING** and **COUNT()**. Once again, we'll use this made-up table of information on pets. 
# 
# ![](https://i.imgur.com/fI5Pvvp.png)
# 
# # COUNT()
# 
# **COUNT()**, as you may have guessed from the name, returns a count of things. If you pass it the name of a column, it will return the number of entries in that column. 
# 
# For instance, if we **SELECT** the **COUNT()** of the `ID` column in the `pets` table, it will return 4, because there are 4 ID's in the table.
# 
# ![](https://i.imgur.com/Eu5HkXq.png)
# 
# **COUNT()** is an example of an **aggregate function**, which takes many values and returns one. (Other examples of aggregate functions include **SUM()**, **AVG()**, **MIN()**, and **MAX()**.)  As you'll notice in the picture above, aggregate functions introduce strange column names (like `f0__`).  Later in this tutorial, you'll learn how to change the name to something more descriptive.
#  
# # GROUP BY
# 
# 
# **GROUP BY** takes the name of one or more columns, and treats all rows with the same value in that column as a single group when you apply aggregate functions like **COUNT()**.
# 
# For example, say we want to know how many of each type of animal we have in the `pets` table. We can use **GROUP BY** to group together rows that have the same value in the `Animal` column, while using **COUNT()** to find out how many ID's we have in each group. 
# 
# ![](https://i.imgur.com/tqE9Eh8.png)
# 
# It returns a table with three rows (one for each distinct animal).  We can see that the `pets` table contains 1 rabbit, 1 dog, and 2 cats.
# 
# # GROUP BY ... HAVING
# 
# **HAVING** is used in combination with **GROUP BY** to ignore groups that don't meet certain criteria. 
# 
# So this query, for example, will only include groups that have more than one ID in them.
# 
# ![](https://i.imgur.com/2ImXfHQ.png)
# 
# Since only one group meets the specified criterion, the query will return a table with only one row. 
# 
# # Example: Which Hacker News comments generated the most discussion?
# 
# Ready to see an example on a real dataset? The Hacker News dataset contains information on stories and comments from the Hacker News social networking site. 
# 
# We'll work with the `comments` table and begin by printing the first few rows.  (_We have hidden the corresponding code. To take a peek, click on the "Code" button below._)

# In[ ]:



from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "hacker_news" dataset
dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# Construct a reference to the "comments" table
table_ref = dataset_ref.table("comments")

# API request - fetch the table
table = client.get_table(table_ref)

# Preview the first five lines of the "comments" table
client.list_rows(table, max_results=5).to_dataframe()


# Let's use the table to see which comments generated the most replies.  Since:
# - the `parent` column indicates the comment that was replied to, and 
# - the `id` column has the unique ID used to identify each comment, 
# 
# we can **GROUP BY** the `parent` column and **COUNT()** the `id` column in order to figure out the number of comments that were made as responses to a specific comment.  (_This might not make sense immediately -- take your time here to ensure that everything is clear!_)
# 
# Furthermore, since we're only interested in popular comments, we'll look at comments with more than ten replies.  So, we'll only return groups **HAVING** more than ten ID's.

# In[ ]:


# Query to select comments that received more than 10 replies
query_popular = """
                SELECT parent, COUNT(id)
                FROM `bigquery-public-data.hacker_news.comments`
                GROUP BY parent
                HAVING COUNT(id) > 10
                """


# Now that our query is ready, let's run it and store the results in a pandas DataFrame: 

# In[ ]:


# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 10 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query_popular, job_config=safe_config)

# API request - run the query, and convert the results to a pandas DataFrame
popular_comments = query_job.to_dataframe()

# Print the first five rows of the DataFrame
popular_comments.head()


# Each row in the `popular_comments` DataFrame corresponds to a comment that received more than ten replies.  For instance, the comment with ID `801208` received `56` replies.
# 
# # Aliasing and other improvements
# 
# A couple hints to make your queries even better:
# - The column resulting from `COUNT(id)` was called `f0__`. That's not a very descriptive name. You can change the name by adding `AS NumPosts` after you specify the aggregation. This is called **aliasing**, and it will be covered in more detail in an upcoming lesson.
# - If you are ever unsure what to put inside the **COUNT()** function, you can do `COUNT(1)` to count the rows in each group. Most people find it especially readable, because we know it's not focusing on other columns. It also scans less data than if supplied column names (making it faster and using less of your data access quota).
# 
# Using these tricks, we can rewrite our query:

# In[ ]:


# Improved version of earlier query, now with aliasing & improved readability
query_improved = """
                 SELECT parent, COUNT(1) AS NumPosts
                 FROM `bigquery-public-data.hacker_news.comments`
                 GROUP BY parent
                 HAVING COUNT(1) > 10
                 """

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query_improved, job_config=safe_config)

# API request - run the query, and convert the results to a pandas DataFrame
improved_df = query_job.to_dataframe()

# Print the first five rows of the DataFrame
improved_df.head()


# Now you have the data you want, and it has descriptive names. That's good style.
# 
# # Note on using **GROUP BY**
# 
# Note that because it tells SQL how to apply aggregate functions (like **COUNT()**), it doesn't make sense to use **GROUP BY** without an aggregate function.  Similarly, if you have any **GROUP BY** clause, then all variables must be passed to either a
# 1. **GROUP BY** command, or
# 2. an aggregation function.
# 
# Consider the query below:
# 
# 

# In[ ]:


query_good = """
             SELECT parent, COUNT(id)
             FROM `bigquery-public-data.hacker_news.comments`
             GROUP BY parent
             """


# Note that there are two variables: `parent` and `id`. 
# - `parent` was passed to a **GROUP BY** command (in `GROUP BY parent`), and 
# - `id` was passed to an aggregate function (in `COUNT(id)`).
# 
# And this query won't work, because the `author` column isn't passed to an aggregate function or a **GROUP BY** clause:

# In[ ]:


query_bad = """
            SELECT author, parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            """


# If make this error, you'll get the error message `SELECT list expression references column (column's name) which is neither grouped nor aggregated at`.
# 
# # Your turn
# 
# These aggregations let you write much more interesting queries. Try it yourself with **[these coding exercises](https://www.kaggle.com/kernels/fork/682058)**.

# ---
# 
# 
# 
# 
# *Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/intro-to-sql/discussion) to chat with other learners.*
