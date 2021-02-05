The following are the steps for executing the bi-stage FS:

1. Enter the original feature set in the dataset variable in the file named filter_feat_select which produces a union of top 150 ranked features using both MI and ReliefF feature selection methods. It gives out a feature file new_covid_feature_red.csv.

2. Now, enter the new_covid_feature_red.csv file in the dataset variable in the main.py file which will apply Dragonfly algorithm on the ranked feature sets and will produce the output metrics and the generate a final reduced feature set as a file named reduced_feature_set.csv
