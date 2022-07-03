# Prediction of survival time of patients with amyotrophic lateral sclerosis based on hydrogen bond graphs
This work is graduate work. I've been doing it for two years, the result and process present in this [Notebook](https://github.com/Voronik1801/BioProject/blob/master/Prediction%20Survival%20Time.ipynb)
# Purpose and objectives
- Development of a method for calculating the characteristics of the graph of hydrogen bonds, which will allow selecting optimal sets of characteristics and optimal calculation methods that give the best prediction result. 

- Construction and investigation of regression models linking the characteristics of the graph of hydrogen bonds associated with ALS with the survival of patients who are carriers of these mutations. Prediction of survival of patients with mutations in SOD1.

- Analysis of the results of the constructed regression model and identification of the sites of the SOD1 enzyme that affect life expectancy.
# Input data
The data contain 182 signs of hydrogen bonds of the enzyme SOD1 for 72 patients.
# Method
It is assumed that the course of the disease affects the structure of the graph, therefore, when compiling training samples, the characteristics of the raf are calculated. A linear regression is constructed for the constructed training sample.
# Results
It turned out to isolate significant subgraphs of the SOD1 enzyme, as well as to improve the RMSE twice.
All the details are [here](https://github.com/Voronik1801/BioProject/blob/master/Prediction%20Survival%20Time.ipynb)
# Author
Voronkina Daria - Graduated student at [NSTU](https://www.nstu.ru/)
