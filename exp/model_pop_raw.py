"""
Experiment summary
------------------
Model a regression of testing based on population, and classify population/testing points with deaths
"""

import sys
sys.path.insert(0, '..')

import matplotlib.pyplot as plt
from utils import data
import os
import sklearn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import json

# ------------ HYPERPARAMETERS -------------
BASE_PATH = '../COVID-19/csse_covid_19_data/'
GREEN = .0001 #deaths / pop
ORANGE = .0005 #deaths / pop
RED = 1 #deaths / pop
# ------------------------------------------

pop_data = os.path.join(
    BASE_PATH, 
    'csse_covid_19_time_series',
    'time_series_covid19_deaths_US.csv')

test_data = os.path.join(
    BASE_PATH,
    'csse_covid_19_daily_reports_us',
    '05-19-2020.csv')

# area_data = os.path.join(
#     BASE_PATH,
#     'areas.csv'
# )

pop_data = data.load_csv_data(pop_data)
test_data = data.load_csv_data(test_data)
# area_data = data.load_csv_data(area_data)
population = []
tests = []
death_proportion = []
deaths_labels = []
# areas = [] #square miles
# densities = [] #store population density here for each state

for state in np.unique(pop_data['Province_State']):
    if state == 'Grand Princess' or state == 'Diamond Princess':
        continue
    sum_pop = 0
    for id_curr_state in range(len(pop_data['Province_State'])):
        if pop_data['Province_State'][id_curr_state] == state:
            sum_pop += pop_data['Population'][id_curr_state]
    population.append(sum_pop)
    for id_curr_state in range(len(test_data['Province_State'])):
        if test_data['Province_State'][id_curr_state] == state:
            n_tested = test_data['People_Tested'][id_curr_state]
    tests.append(n_tested)
    for id_curr_state in range(len(test_data['Province_State'])):
        if test_data['Province_State'][id_curr_state] == state:
            n_deaths = test_data['Deaths'][id_curr_state]
    death_proportion.append(n_deaths / sum_pop)
    # for id_curr_state in range(len(area_data['State'])):
    #     if area_data['State'][id_curr_state] == state:
    #         area = area_data['TotalArea'][id_curr_state]
    # areas.append(area)


# for idx in range(len(areas)):
#     densities.append(population[idx] / areas[idx])

# print(death_proportion)
# print('highest prop = ', np.amax(death_proportion))

for num in death_proportion:
    if num < GREEN:
        deaths_labels.append('green')
    elif num >= GREEN and num < ORANGE:
        deaths_labels.append('orange')
    elif num >= ORANGE and num < RED:
        deaths_labels.append('red')
# print(deaths)

plt.figure(figsize=(6,4))
plt.scatter(population, tests, color = deaths_labels)
plt.xlabel('Population')
plt.ylabel('# Tested')
# plt.savefig('pop_tests_scatter.png')


#regression
p = np.poly1d(np.polyfit(population, tests, 6))
x_range = np.linspace(0, 4e7)
line = p(x_range)
plt.plot(x_range, line)
plt.title('Raw Population')
# enter population to predict tests for
pop_to_predict = 1.5e7
testing_prediction = p(pop_to_predict)
print('Number of tested individuals expected = ', testing_prediction)

#knn
#combine points into features array
features = np.ones((len(population), 2))
for i in range(len(population)):
    features[i, 0] = population[i]
    features[i, 1] = tests[i]
#assign values to color labels
num_labels = []
for color in deaths_labels:
    if color == 'green':
        num_labels.append(1)
    elif color == 'orange':
        num_labels.append(3)
    elif color == 'red':
        num_labels.append(6)
knn = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
knn.fit(features, num_labels)
# enter population and tests to predict death level for
input_pop_tests = [pop_to_predict, testing_prediction]
idx_knearest = knn.kneighbors([input_pop_tests], return_distance=False)[0]
knearest_labels = []
for idx in idx_knearest:
    knearest_labels.append(num_labels[idx])

# print(idx_knearest)
# print(num_labels)
# print(features[idx_knearest])

# labels are weighted for severity
avg_label = np.mean(knearest_labels)
# print(avg_label)
# print(knearest_labels)
# get closest value out of all labels to the average label, rounded up for extra precaution
dist_avg = []
for l in num_labels:
    dist_avg.append(np.absolute(l - avg_label))
predicted_label = np.amax(num_labels[np.argmin(dist_avg)])

# print(predicted_label)
if predicted_label == 1:
    prediction_death = 'green'
elif predicted_label == 3:
    prediction_death = 'orange'
elif predicted_label == 6:
    prediction_death = 'red'


    
print('Given population ' + str(input_pop_tests[0]) + ' and number tested ' + str(input_pop_tests[1]) + ', the predicted death level = ' + str(prediction_death))

#plot new pop, test, death level
plt.scatter(input_pop_tests[0], input_pop_tests[1], color=prediction_death, marker='^')


plt.savefig('model_pop_raw.png')

