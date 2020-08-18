# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Feature Engineering
#
# In this notebook we're going to generate features for our link prediction classifier.

# +
from neo4j import GraphDatabase

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# tag::imports[]
from sklearn.ensemble import RandomForestClassifier
# end::imports[]

# +
bolt_uri = "bolt://link-prediction-neo4j"
driver = GraphDatabase.driver(bolt_uri, auth=("neo4j", "admin"))

print(driver.address)
# -

# We can create our classifier with the following code:

# +
# Load the CSV files saved in the train/test notebook

df_train_under = pd.read_csv("data/df_train_under.csv")
df_test_under = pd.read_csv("data/df_test_under.csv")
# -

df_train_under.sample(5)

df_test_under.sample(5)


# # Generating graphy features
#
# We’ll start by creating a simple model that tries to predict whether two authors will have a future collaboration based on features extracted from common authors, preferential attachment, and the total union of neighbors.
#
# The following function computes each of these measures for pairs of nodes:

# tag::graphy-features[]
def apply_graphy_features(data, rel_type):
    query = """
    UNWIND $pairs AS pair
    MATCH (p1) WHERE id(p1) = pair.node1
    MATCH (p2) WHERE id(p2) = pair.node2
    RETURN pair.node1 AS node1,
           pair.node2 AS node2,
           gds.alpha.linkprediction.commonNeighbors(p1, p2, {
             relationshipQuery: $relType}) AS cn,
           gds.alpha.linkprediction.preferentialAttachment(p1, p2, {
             relationshipQuery: $relType}) AS pa,
           gds.alpha.linkprediction.totalNeighbors(p1, p2, {
             relationshipQuery: $relType}) AS tn
    """
    pairs = [{"node1": node1, "node2": node2}  for node1,node2 in data[["node1", "node2"]].values.tolist()]
    
    with driver.session(database="neo4j") as session:
        result = session.run(query, {"pairs": pairs, "relType": rel_type})
        features = pd.DataFrame([dict(record) for record in result])    
    return pd.merge(data, features, on = ["node1", "node2"])
# end::graphy-features[]


# Let's apply the function to our training DataFrame:

# tag::apply-graphy-features[]
df_train_under = apply_graphy_features(df_train_under, "CO_AUTHOR_EARLY")
df_test_under = apply_graphy_features(df_test_under, "CO_AUTHOR")
# end::apply-graphy-features[]

# Now we're going to add some new features that are generated from graph algorithms.
#
# ## Triangles and The Clustering Coefficient
#
# We'll start by running the [triangle count](clusteringCoefficientProperty) algorithm over our test and train sub graphs. This algorithm will return the number of triangles that each node forms, as well as each node's clustering coefficient. The clustering coefficient of a node indicates the likelihood that its neighbours are also connected.

# +
query = """
CALL gds.triangleCount.write({
  nodeProjection: 'Author',
  relationshipProjection: {
    CO_AUTHOR_EARLY: {
      type: 'CO_AUTHOR_EARLY',
      orientation: 'UNDIRECTED'
    }
  },
  writeProperty: 'trianglesTrain'
});
"""

with driver.session(database="neo4j") as session:
    result = session.run(query)
    df = pd.DataFrame([dict(record) for record in result])
df

# +
query = """
CALL gds.triangleCount.write({
  nodeProjection: 'Author',
  relationshipProjection: {
    CO_AUTHOR: {
      type: 'CO_AUTHOR',
      orientation: 'UNDIRECTED'
    }
  },
  writeProperty: 'trianglesTest'
});
"""

with driver.session(database="neo4j") as session:
    result = session.run(query)
    df = pd.DataFrame([dict(record) for record in result])
df    

# +
query = """
CALL gds.localClusteringCoefficient.write({
  nodeProjection: 'Author',
  relationshipProjection: {
    CO_AUTHOR_EARLY: {
      type: 'CO_AUTHOR',
      orientation: 'UNDIRECTED'
    }
  },
  writeProperty: 'coefficientTest'
});
"""

with driver.session(database="neo4j") as session:
    result = session.run(query)
    df = pd.DataFrame([dict(record) for record in result])
df
# -

The following function will add these features to our train and test DataFrames:


def apply_triangles_features(data, triangles_prop, coefficient_prop):
    query = """
    UNWIND $pairs AS pair
    MATCH (p1) WHERE id(p1) = pair.node1
    MATCH (p2) WHERE id(p2) = pair.node2
    RETURN pair.node1 AS node1,
    pair.node2 AS node2,
    apoc.coll.min([p1[$trianglesProp], p2[$trianglesProp]]) AS minTriangles,
    apoc.coll.max([p1[$trianglesProp], p2[$trianglesProp]]) AS maxTriangles,
    apoc.coll.min([p1[$coefficientProp], p2[$coefficientProp]]) AS minCoefficient,
    apoc.coll.max([p1[$coefficientProp], p2[$coefficientProp]]) AS maxCoefficient
    """    
    pairs = [{"node1": node1, "node2": node2}  for node1,node2 in data[["node1", "node2"]].values.tolist()]    
    params = {
    "pairs": pairs,
    "trianglesProp": triangles_prop,
    "coefficientProp": coefficient_prop
    }

    with driver.session(database="neo4j") as session:
        result = session.run(query, params)
        features = pd.DataFrame([dict(record) for record in result])       
    
    return pd.merge(data, features, on = ["node1", "node2"])


df_train_under = apply_triangles_features(df_train_under, "trianglesTrain", "coefficientTrain")
df_test_under = apply_triangles_features(df_test_under, "trianglesTest", "coefficientTest")

df_train_under.sample(5)

df_test_under.sample(5)

# ## Community Detection
#
# Community detection algorithms evaluate how a group is clustered or partitioned. Nodes are considered more similar to nodes that fall in their community than to nodes in other communities.
#
# We'll run two community detection algorithms over the train and test sub graphs - Label Propagation and Louvain. First up, Label Propagation: 

# +
query = """
CALL gds.labelPropagation.write({
  nodeProjection: "Author",
  relationshipProjection: {
    CO_AUTHOR_EARLY: {
      type: 'CO_AUTHOR_EARLY',
      orientation: 'UNDIRECTED'
    }
  },
  writeProperty: "partitionTrain"
});
"""

with driver.session(database="neo4j") as session:
    result = session.run(query)
    df = pd.DataFrame([dict(record) for record in result])
df    

# +
query = """
CALL gds.labelPropagation.write({
  nodeProjection: "Author",
  relationshipProjection: {
    CO_AUTHOR: {
      type: 'CO_AUTHOR',
      orientation: 'UNDIRECTED'
    }
  },
  writeProperty: "partitionTest"
});
"""

with driver.session(database="neo4j") as session:
    result = session.run(query)
    df = pd.DataFrame([dict(record) for record in result])
df    

# +
query = """
CALL gds.louvain.stream({
  nodeProjection: 'Author',
  relationshipProjection: {
    CO_AUTHOR_EARLY: {
      type: 'CO_AUTHOR_EARLY',
      orientation: 'UNDIRECTED'
    }
  },
  includeIntermediateCommunities: true
})
YIELD nodeId, communityId, intermediateCommunityIds
WITH gds.util.asNode(nodeId) AS node, intermediateCommunityIds[0] AS smallestCommunity
SET node.louvainTrain = smallestCommunity;
"""

with driver.session(database="neo4j") as session:
    display(session.run(query).consume().counters)

# +
query = """
CALL gds.louvain.stream({
  nodeProjection: 'Author',
  relationshipProjection: {
    CO_AUTHOR: {
      type: 'CO_AUTHOR',
      orientation: 'UNDIRECTED'
    }
  },
  includeIntermediateCommunities: true
})
YIELD nodeId, communityId, intermediateCommunityIds
WITH gds.util.asNode(nodeId) AS node, intermediateCommunityIds[0] AS smallestCommunity
SET node.louvainTest = smallestCommunity;
"""

with driver.session(database="neo4j") as session:
    display(session.run(query).consume().counters)


# -

def apply_community_features(data, partition_prop, louvain_prop):
    query = """
    UNWIND $pairs AS pair
    MATCH (p1) WHERE id(p1) = pair.node1
    MATCH (p2) WHERE id(p2) = pair.node2
    RETURN pair.node1 AS node1,
    pair.node2 AS node2,
    gds.alpha.linkprediction.sameCommunity(p1, p2, $partitionProp) AS sp,    
    gds.alpha.linkprediction.sameCommunity(p1, p2, $louvainProp) AS sl
    """
    pairs = [{"node1": node1, "node2": node2}  for node1,node2 in data[["node1", "node2"]].values.tolist()]
    params = {
    "pairs": pairs,
    "partitionProp": partition_prop,
    "louvainProp": louvain_prop
    }
    
    with driver.session(database="neo4j") as session:
        result = session.run(query, params)
        features = pd.DataFrame([dict(record) for record in result])
    
    return pd.merge(data, features, on = ["node1", "node2"])


df_train_under = apply_community_features(df_train_under, "partitionTrain", "louvainTrain")
df_test_under = apply_community_features(df_test_under, "partitionTest", "louvainTest")

# tag::train-after-features[]
df_train_under.sample(5)
# end::train-after-features[]

# tag::test-after-features[]
df_test_under.sample(5)
# end::test-after-features[]

# +
# Save our DataFrames to CSV files for use in the next notebook

df_train_under.to_csv("data/df_train_under_all.csv", index=False)
df_test_under.to_csv("data/df_test_under_all.csv", index=False)

# +
# tag::train-model[]
columns = [
    "cn", "pa", "tn", # graph features
    "minTriangles", "maxTriangles", "minCoefficient", "maxCoefficient", # triangle features  
    "sp", "sl" # community features
]

X = df_train_under[columns]
y = df_train_under["label"]
classifier.fit(X, y)
# end::train-model[]

# +
# tag::evaluation-imports[]
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
# end::evaluation-imports[]

# tag::evaluation-functions[]
def evaluate_model(predictions, actual):
    return pd.DataFrame({
        "Measure": ["Accuracy", "Precision", "Recall"],
        "Score": [accuracy_score(actual, predictions), 
                  precision_score(actual, predictions), 
                  recall_score(actual, predictions)]
    })

def feature_importance(columns, classifier):        
    display("Feature Importance")
    df = pd.DataFrame({
        "Feature": columns,
        "Importance": classifier.feature_importances_
    })
    df = df.sort_values("Importance", ascending=False)    
    ax = df.plot(kind='bar', x='Feature', y='Importance', legend=None)
    ax.xaxis.set_label_text("")
    plt.tight_layout()
    plt.show()
# end::evaluation-functions[]


# +
# tag::test-model[]
predictions = classifier.predict(df_test_under[columns])
y_test = df_test_under["label"]

display(evaluate_model(predictions, y_test))
feature_importance(columns, classifier)
# end::test-model[]
