import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

file_path=os.path.join("datasets","housing","housing.csv")
housing=pd.read_csv(file_path)

y=housing['population']
x=housing['ocean_proximity']
plt.bar(x,y)
plt.get_current_fig_manager().set_window_title('Bargraph of population and ocean proximity')


housing.hist()
plt.get_current_fig_manager().set_window_title('Histogram of complete dataset')
plt.show()

housing["median_income"].hist()
plt.get_current_fig_manager().set_window_title('Histogram of median income')
plt.show()

housing["income_cat"]=pd.cut(housing["median_income"],bins=[0.,1.5,3.,4.5,6,np.inf],labels=[1,2,3,4,5])
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=21)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]

housing["income_cat"].hist()
plt.get_current_fig_manager().set_window_title('Histogram of income category')


print("\nPercentage of income category in complete housing database ranging from 1-5\n")
print(housing["income_cat"].value_counts()/len(housing))
print("\nPercentage of income category stratiefied sampling train set database ranging from 1-5\n")
print(strat_train_set["income_cat"].value_counts()/len(strat_train_set))
print("\nPercentage of income category stratiefied sampling test set database ranging from 1-5\n")
print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))

housing.plot(kind="scatter",x="longitude",y="latitude")
plt.get_current_fig_manager().set_window_title('Scatterplot without alpha value')

housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1)
plt.get_current_fig_manager().set_window_title('Scatterplot with alpha value')


housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,s=housing["population"]/100,label="population",figsize=(10,7),c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)
plt.get_current_fig_manager().set_window_title('Scatterplot using cmap')
plt.show()

#corr_matrix=strat_train_set.corr(method="pearson",numeric_only=True)
pd.plotting.scatter_matrix(housing)
plt.get_current_fig_manager().set_window_title('Correlation of data set')
plt.show()

housing["rooms_per_household"]=housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"]=housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix=housing.corr(method="pearson",numeric_only=True)

print(corr_matrix["median_house_value"].sort_values(ascending=False))





