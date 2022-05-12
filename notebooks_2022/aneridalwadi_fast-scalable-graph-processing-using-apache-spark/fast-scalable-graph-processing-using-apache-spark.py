#!/usr/bin/env python
# coding: utf-8

# ## Problems with NetworkX
# 
# <img
#   src="https://networkx.org/_static/networkx_logo.svg"
#   alt="nx"
#   width="300"
#   height="300"
# />
# 
# [NetworkX](https://networkx.org/) is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
# 
# NetworkX has an easy and intuitive API but it comes with a major flaw. All the data in NetworkX is stored in-memory. This will fill up the RAM pretty fast and thus lead to out-of-memory (OOM) errors when working with large graphs. Another major problem with NetworkX is that it does not allow for ad-hoc graph querying on edges and vertices. For example, if one wanted to find all the edges connected to a particular set of nodes, we would have to resort to manual looping which is very slow when working with large data.
# 
# We can avoid this two problems entirely by using some solution which accounts for both the above mentioned problems. We can use [Apache Spark](https://spark.apache.org/) for this. Spark is a unified analytics engine for large-scale data processing. It provides high-level APIs in Scala, Java, Python, and R, and an optimized engine that supports general computation graphs for data analysis. It also supports a rich set of higher-level tools including Spark SQL for SQL and DataFrames, pandas API on Spark for pandas workloads, MLlib for machine learning, GraphX for graph processing, and Structured Streaming for stream processing.
# 
# <img
#   src="https://spark.apache.org/images/spark-logo.png"
#   alt="spark"
#   width="180"
#   height="180"
#   style="fill: black"
# />
# 
# This notebook is a tutorial on how to export graph data from NetworkX and import it into Apache Spark.

# In[ ]:


import networkx as nx


# We can use an already existing NetworkX graph, create a new graph, or generate a random graph. This method works independently to any method by which graph was created.
# 
# For the purpose of this tutorial, we will generate a random graph. The full list of [graph generators](https://networkx.org/documentation/stable/reference/generators.html) can be found in the documentation.

# In[ ]:


g = nx.caveman_graph(120, 100)


# Spark requires a DataFrame to work with. We can create a pandas DataFrame from NetworkX directly and then read it into Spark.
# Once we have our graph/network, we use the [`nx.to_pandas_edgelist`](https://networkx.org/documentation/stable/reference/generated/networkx.convert_matrix.from_pandas_edgelist.html) function. The function returns a `pd.DataFrame` object.

# In[ ]:


df = nx.to_pandas_edgelist(g)


# In[ ]:


df


# Depending on the type of data, we can now save it in multiple formats like CSV, JSON, and Apache Parquet.

# In[ ]:


df.to_csv("graph.csv", index=False)


# ## Apache Spark
# 
# To install Apache Spark, use the following command

# In[ ]:


get_ipython().system('pip install pyspark')


# We will now need to initialize a Spark session before we can work with the data.

# In[ ]:


import pyspark.sql as sql


# In[ ]:


spark = (
    sql.SparkSession.builder.config("spark.driver.memory", "8g")
    .config("spark.sql.execution.arrow.pyspark.enable", "true")
    .config("spark.driver.maxResultSize", "2g")
    .getOrCreate()
)


# In[ ]:


df = spark.read.csv("graph.csv", inferSchema=True, header=True)
df.createOrReplaceTempView("graph")


# Now we have access to the entire data without ever having to worry about running into OOM errors.
# The reason this works is Spark is designed to work with the data directly on the disc without having
# to read it into memory. It is able to query the data in distributed way and can scale for even petabytes of data.
# 
# We can now query the data as if it is a table in SQL.

# In[ ]:


# Count of edges from node 15
spark.sql(
    """
select
  count(target)
from
  graph
where
  source = 15
    """
).collect()


# In[ ]:


# List of nodes connect to node 35
spark.sql(
    """
select
  source
from
  graph
where
  target = 35
    """
).collect()


# In[ ]:


# List of nodes connected with more than 20 nodes and the count of connections
spark.sql(
    """
select
  source,
  count(target) cnt
from
  graph
group by
  source
having
  cnt > 20
limit
  20
    """
).collect()


# The documentation of Apache Spark contains the full list of functions available for use. Refer to this notebook for a reference on how to use Apache Spark for large scale data analysis - [Exploratory Data Analysis - Taylor Swift](https://www.kaggle.com/code/aneridalwadi/exploratory-data-analysis-taylor-swift)

# ## cuGraph
# 
# 
# Spark resorts to using SQL for data manipulation which can often turn out to be unintuitive and hard to use and NetworkX runs only on CPU.
# This problem is solved by using `cuGraph` library created by NVIDIA which aims to do data processing and analysis on GPUs directly. It has an API similar to NetworkX and supposed to work as a drop-in replacement. It won't save us from OOM errors because now it will use GPU memory instead but it can speed up complicated graph calculations by orders of magnitude.
# 
# <img
#   src="https://github.com/rapidsai/cugraph/blob/main/img/rapids_logo.png?raw=true"
#   alt="rapids"
#   width="200"
#   height="200"
# />

# In[ ]:


get_ipython().run_line_magic('conda', 'install -c nvidia -c rapidsai -c numba -c conda-forge cugraph')


# In[ ]:


import cudf
import cugraph

# read data into a cuDF DataFrame using read_csv
gdf = cudf.read_csv("graph.csv", dtype=["int32", "int32"])

# We now have data as edge pairs
# create a Graph using the source (src) and destination (dst) vertex pairs
G = cugraph.Graph()
G.from_cudf_edgelist(gdf, source="source", destination="target")

# Let's now get the PageRank score of each vertex by calling cugraph.pagerank
cugraph.pagerank(G)

