import pandas as pd
import numpy as np
from patsy.highlevel import dmatrices, dmatrix
import statsmodels.api as sm

# Say whether you want to run the 'normal' code (using just local authority results) or 'Adjusted' (where you treat the Edinburgh constituencies as local authorities since you know the results there)

version='Adjusted'
edinburgh_dict={'Edinburgh East':'S14000022','Edinburgh North and Leith':'S14000023','Edinburgh South':'S14000024','Edinburgh South West':'S14000025','Edinburgh West':'S14000026'}

# Read in look up table from Census Output Areas to Local Authorities and Westminster Constituencies

index_table = pd.read_csv('OA_TO_HIGHER_AREAS.csv')

# Restrict the index table to just be COA, Local Authority and Westminster Const

index_table = index_table.reindex(columns=['OutputArea2011Code','CouncilArea2011Code','UKParliamentaryConstituency2005Code'])

# Create a new column showing which 'intersection' between LA and Westminster Const a CAO is in, and then make CAO the index

if version=='Adjusted':
    index_table.loc[index_table['UKParliamentaryConstituency2005Code'].isin(list(edinburgh_dict.values())),'CouncilArea2011Code']=index_table.loc[index_table['UKParliamentaryConstituency2005Code'].isin(list(edinburgh_dict.values())),'UKParliamentaryConstituency2005Code']
else:
    pass

index_table.to_csv('indextable.csv')

index_table.loc[:,'Intersection']=index_table.CouncilArea2011Code + index_table.UKParliamentaryConstituency2005Code
index_table.set_index('OutputArea2011Code',inplace=True) 

# Read demographic data of interest into a dictionary of dataframes for all COAs - Age, Country of Birth, Owner Occupiers

table_dict={'KS102SC':'age','KS204SC':'birth_country','KS611SC':'occupation','KS601SC':'unemployed'}

df_dict= {value: pd.read_csv(key+'.csv') for key, value in table_dict.items()}

# Separately read in benefit claimant data, since this is in a different format

df_benefits=pd.read_csv('benefit_claims_by_coa_aug_2014_edited.csv')
df_benefits.set_index('OutputArea2011Code',inplace=True)

# Rename the first column of all dataframes 'OutputArea2011Code'
# Replace '-' values with 0 (so they can be manipulated as numbers)

df_dict= {key: value.rename(columns={'Unnamed: 0':'OutputArea2011Code'}) for (key, value) in df_dict.items()}
df_dict= {key: value.replace('-',0) for (key, value) in df_dict.items()}

# Create a starting dataframe which is just the list of COA codes plus total population, for later
# Set index of this and all dictionary dataframes to be the output area code
# Make get rid of thousand comma separators and make all values numbers

df_population=df_dict['age'].loc[:,('OutputArea2011Code','All people')]
df_population.set_index('OutputArea2011Code',inplace=True)
df_population=df_population.astype(str).apply(lambda s: s.str.replace(',','')).astype(float)
df_dict= {key: value.set_index('OutputArea2011Code') for (key, value) in df_dict.items()}
df_dict= {key: value.astype(str).apply(lambda s: s.str.replace(',','')).astype(float) for (key, value) in df_dict.items()}

# Manipulate each dataframe to get only the columns you want

df_dict['age'].loc[:,'16 to 19']=df_dict['age'].loc[:,'16 to 17']+df_dict['age'].loc[:,'18 to 19']
df_dict['age'].loc[:,'65 and over']=df_dict['age'].loc[:,'65 to 74']+df_dict['age'].loc[:,'75 to 84']+df_dict['age'].loc[:,'85 to 89']+df_dict['age'].loc[:,'90 and over']
df_dict['age']=df_dict['age'].loc[:,('16 to 19','20 to 24','25 to 29','30 to 44','45 to 59','60 to 64','65 and over')]

df_dict['birth_country'].loc[:,'RUK']=df_dict['birth_country'].loc[:,'England'] + df_dict['birth_country'].loc[:,'Wales'] + df_dict['birth_country'].loc[:,'Northern Ireland']
df_dict['birth_country']=df_dict['birth_country'].loc[:,('Scotland','RUK')]

df_dict['occupation'].loc[:,'managerial_and_professional']=df_dict['occupation'].loc[:,'1. Higher managerial, administrative and professional occupations: Total']+df_dict['occupation'].loc[:,'2. Lower managerial and professional occupations']
df_dict['occupation']=df_dict['occupation'].loc[:,('All people aged 16 to 74','managerial_and_professional')]

df_dict['unemployed']=df_dict['unemployed'].loc[:,('Economically active: Unemployed')]

# Create single dataframe with all variables of interest in

df=df_population.join(df_dict.values(),how='left')

# Read the local authority and Westminster constituency codes into the dataframe

df=df.join(index_table,how='left')

# Create tables aggregating all measures for each local authority, each intersection and each constituency (index for both now becomes la or intersection code or UK Parliament Code)

df_las=df.groupby('CouncilArea2011Code').sum()
df_intersection=df.groupby('Intersection').sum()
df_const=df.groupby('UKParliamentaryConstituency2005Code').sum()

# Translate numbers into percentages - of all people for most variables, and all aged 16-74 for managerial / professional and unemployed

df_las.loc[:,~df_las.columns.isin(['All people','All people aged 16 to 74','managerial_and_professional','Economically active: Unemployed'])]=df_las.loc[:,~df_las.columns.isin(['All people','All people aged 16 to 74','managerial_and_professional','Economically active: Unemployed'])].divide(df_las['All people'],axis=0)
df_las.loc[:,'managerial_and_professional']=df_las.loc[:,'managerial_and_professional'].divide(df_las['All people aged 16 to 74'],axis=0)
df_las.loc[:,'Economically active: Unemployed']=df_las.loc[:,'Economically active: Unemployed'].divide(df_las['All people aged 16 to 74'],axis=0)
df_las.drop(columns=['All people aged 16 to 74'],inplace=True)

df_intersection.loc[:,~df_intersection.columns.isin(['All people','All people aged 16 to 74','managerial_and_professional','Economically active: Unemployed'])]=df_intersection.loc[:,~df_intersection.columns.isin(['All people','All people aged 16 to 74','managerial_and_professional','Economically active: Unemployed'])].divide(df_intersection['All people'],axis=0)
df_intersection.loc[:,'managerial_and_professional']=df_intersection.loc[:,'managerial_and_professional'].divide(df_intersection['All people aged 16 to 74'],axis=0)
df_intersection.loc[:,'Economically active: Unemployed']=df_intersection.loc[:,'Economically active: Unemployed'].divide(df_intersection['All people aged 16 to 74'],axis=0)
df_intersection.drop(columns=['All people aged 16 to 74'],inplace=True)

df_const.loc[:,~df_const.columns.isin(['All people','All people aged 16 to 74','managerial_and_professional','Economically active: Unemployed'])]=df_const.loc[:,~df_const.columns.isin(['All people','All people aged 16 to 74','managerial_and_professional','Economically active: Unemployed'])].divide(df_const['All people'],axis=0)
df_const.loc[:,'managerial_and_professional']=df_const.loc[:,'managerial_and_professional'].divide(df_const['All people aged 16 to 74'],axis=0)
df_const.loc[:,'Economically active: Unemployed']=df_const.loc[:,'Economically active: Unemployed'].divide(df_const['All people aged 16 to 74'],axis=0)
df_const.drop(columns=['All people aged 16 to 74'],inplace=True)

# Read in 2014 Scottish Referendum results by local authority (including mapping between codes and names)
# Rename the Code column to be 'CouncilArea2011Code'
# Set CouncilArea2011Code to be the key to prepare for the join
# Convert all numbers from strings to floats
# Read in raw local authority results or amended file with Edinburgh constituency results added, depending on which version of the model you want to run

if version=='Adjusted':
    results_2014= pd.read_csv('ScotVote4.csv')
else:
    results_2014= pd.read_csv('ScotVote3.csv')

results_2014.rename(columns={'Code': 'CouncilArea2011Code'},inplace=True)
results_2014.set_index('CouncilArea2011Code',inplace=True)
results_2014.loc[:,results_2014.columns!='Council']=results_2014.loc[:,results_2014.columns!='Council'].astype(str).apply(lambda s: s.str.replace(',','')).astype(float)

# Read the results into the local authority dataframe

df_las=df_las.join(results_2014.loc[:,('Yes','No')], how='left')

# Write the form of the regression for the Yes vote and the No vote - i.e. Yes ~ Q("All people") + Q("16 to 19") + etc

expr = 'Q("' + ('") + Q("').join(list(df_las.columns[~df_las.columns.isin(['Yes','No'])])) + '")'
yes_expr='Yes ~ '+ expr
no_expr='No ~ '+ expr

# Run the regression
y_train_yes, X_train_yes = dmatrices(yes_expr, df_las, return_type='dataframe')
poisson_training_results_yes = sm.GLM(y_train_yes, X_train_yes, family=sm.families.Poisson()).fit()

y_train_no, X_train_no = dmatrices(no_expr, df_las, return_type='dataframe')
poisson_training_results_no = sm.GLM(y_train_no, X_train_no, family=sm.families.Poisson()).fit()

# Evaluate the regression

print(poisson_training_results_yes.summary())
print(poisson_training_results_no.summary())

# Then use the model to predict results for Intersections

X_test_yes = dmatrix(expr, df_intersection, return_type='dataframe')
poisson_predictions_yes = poisson_training_results_yes.predict(X_test_yes)

X_test_no = dmatrix(expr, df_intersection, return_type='dataframe')
poisson_predictions_no = poisson_training_results_no.predict(X_test_no)

# And read those results into the intersection dataframe
df_intersection['predicted_yes']=poisson_predictions_yes
df_intersection['predicted_no']=poisson_predictions_no

# Create two new columns in the intersection dataframe, showing the code for la and constituency

intersection_index=index_table.drop_duplicates(subset=['Intersection']).set_index('Intersection')
df_intersection=df_intersection.join(intersection_index.loc[:,('CouncilArea2011Code','UKParliamentaryConstituency2005Code')],how='left')

# Calculate predicted results by local authority, and read these into the las table along with the name of the local authority

df_las_predicted=df_intersection.groupby('CouncilArea2011Code')[['predicted_yes','predicted_no']].sum()
df_las=df_las.join(df_las_predicted,how='left')
df_las=df_las.join(results_2014['Council'],how='left')

# Find mean absolute error of predicted Yes vote for local authorities
df_las['yes_perc']=df_las['Yes']/(df_las['Yes']+df_las['No'])
df_las['predicted_yes_perc']=df_las['predicted_yes']/(df_las['predicted_yes']+df_las['predicted_no'])
df_las['error']=abs(df_las['yes_perc']-df_las['predicted_yes_perc'])

print('Mean absolute error: '+str(df_las['error'].mean()))

# Then calculate the scaling factor for each local authority - how much you need to scale the predicted votes in each intersection to get the right result at local authority level

df_las['yes_scale']=df_las['Yes']/df_las['predicted_yes']
df_las['no_scale']=df_las['No']/df_las['predicted_no']

# Read these scaling factors back into the intersections dataframe and use them to calculate predicted results by Westminster constituency

df_intersection=df_intersection.set_index('CouncilArea2011Code').join(df_las.loc[:,('yes_scale','no_scale')],how='left')
df_intersection['scaled_predicted_yes']=df_intersection['predicted_yes']*df_intersection['yes_scale']
df_intersection['scaled_predicted_no']=df_intersection['predicted_no']*df_intersection['no_scale']

# Calculate results by Westminster constituency

df_const=df_const.join(df_intersection.groupby('UKParliamentaryConstituency2005Code')[['scaled_predicted_yes','scaled_predicted_no']].sum(),how='left')

# Read in Westminster constituency codes to names mapping, rename code column and set as the index

const_codes=pd.read_csv('const_codes.csv')
const_codes.rename(columns={'PCON18CD':'UKParliamentaryConstituency2005Code','PCON18NM':'constituency_name'},inplace=True)
const_codes.set_index('UKParliamentaryConstituency2005Code',inplace=True)

# Read in constituency names to table

df_const=df_const.join(const_codes['constituency_name'],how='left')

# And turn results into percentages

df_const['yes_perc']=df_const['scaled_predicted_yes']/(df_const['scaled_predicted_yes']+df_const['scaled_predicted_no'])
df_const['no_perc']=df_const['scaled_predicted_no']/(df_const['scaled_predicted_yes']+df_const['scaled_predicted_no'])

# Save output to file
df_const.to_csv('predicted_indy_ref_2014_votes.csv')
