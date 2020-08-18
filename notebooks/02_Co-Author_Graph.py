# ---
# jupyter:
#   jupytext:
#     formats: md,py:light
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

# # Building a co-author graph
#
# In this notebook we're going to build an inferred graph of co-authors based on people collaborating on the same papers. We're also going to store a property on the relationship indicating the year of their first collaboration.

# tag::imports[]
from neo4j import GraphDatabase
# end::imports[]

# tag::driver[]
driver = GraphDatabase.driver("bolt://link-prediction-neo4j", auth=("neo4j", "admin"))        
print(driver.address) 
# end::driver[]
