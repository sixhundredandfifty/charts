import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Read in seat data

bes=pd.read_csv('bes_2019.csv')

# Manipulate data so you just have 632 rows, 1 for each seat with a column for each relevant demographic variable. Replace NA with 0.

df=bes.loc[:,('c11Age18to19','c11Age20to24','c11Age25to29','c11Age30to44','c11Age45to59','c11Age60to64','c11Age65to74','c11Age75to84','c11Age85to89','c11Age90plus','c11NSSECHigherManager','c11NSSECHigherProfessional','c11NSSECLowerManager','c11NSSECIntermediate','c11NSSECSmallEmployer','c11NSSECLowerSupervisor','c11NSSECSemiRoutine','c11NSSECRoutine','c11QualNone','c11QualLevel1','c11QualLevel2','c11QualApprentice','c11QualLevel3','c11QualLevel4','c11HouseOwned','c11HouseOutright')]
df=df.fillna(0)

# Create a numpy array of that dataframe

df_array=df.values

# Initialise the algorithm, and set the number of clusters

kmeans=KMeans(n_clusters=5)

# Get the algorithm to break seats into clusters

kmeans.fit(df_array)

# Store the names of the clusters into a new list

listofclusters=kmeans.labels_

# Join that list of clusters to the list of seats

seat_categories=pd.DataFrame({'ONSConstID': bes.ONSConstID,'cluster':listofclusters})

# Add categories into bes

bes=pd.merge(bes,seat_categories,on='ONSConstID')

# Save seat_categories to csv

seat_categories=pd.DataFrame({'ONSConstID': bes.ONSConstID,'cluster_demo':listofclusters})

seat_categories.to_csv('seat_categories_BES_demographics.csv')
