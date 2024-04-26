import Linear_Regression
import Ridge_Regression
import Lasso_Regression
import Polynomial_Regression
import decison_tree
import Random_Forest
import XGBoost
import LightGBM
import CatBoost
import Neural_Networks
import KNN
import Bayesian_Linear_Regression

import matplotlib.pyplot as plt

# LR_R2, LR_MSE, LR_MAE = LR_run()
# RR_R2, RR_MSE, RR_MAE = RR_run()
# Lasso_R2, Lasso_MSE, Lasso_MAE = Lasso_run()
# PR_R2, PR_MSE, PR_MAE = PR_run()
# DT_R2, DT_MSE, DT_MAE = DT_run()
# RF_R2, RF_MSE, RF_MAE = RF_run()
# XGBoost_R2, XGBoost_MSE, XGBoost_MAE = XGBoost_run()
# LightGBM_R2, LightGBM_MSE, LightGBM_MAE = LightGBM_run()
# CatBoost_R2, CatBoost_MSE, CatBoost_MAE = CatBoost_run()
# NN_R2, NN_MSE, NN_MAE = NN_run()
# KNN_R2, KNN_MSE, KNN_MAE = KNN_run()
# Bayesion_R2, Bayesion_MSE, Bayesion_MAE = Bayesion_run()


R2_h = []
MSE_h = []
MAE_h = []

R2, MSE, MAE = Linear_Regression.run()
R2_h.append(R2)
MSE_h.append(MSE)
MAE_h.append(MAE)

R2, MSE, MAE = Ridge_Regression.run()
R2_h.append(R2)
MSE_h.append(MSE)
MAE_h.append(MAE)

R2, MSE, MAE = Lasso_Regression.run()
R2_h.append(R2)
MSE_h.append(MSE)
MAE_h.append(MAE)

R2, MSE, MAE = Polynomial_Regression.run()
R2_h.append(R2)
MSE_h.append(MSE)
MAE_h.append(MAE)

R2, MSE, MAE = decison_tree.run()
R2_h.append(R2)
MSE_h.append(MSE)
MAE_h.append(MAE)

R2, MSE, MAE = Random_Forest.run()
R2_h.append(R2)
MSE_h.append(MSE)
MAE_h.append(MAE)

R2, MSE, MAE = XGBoost.run()
R2_h.append(R2)
MSE_h.append(MSE)
MAE_h.append(MAE)

R2, MSE, MAE = LightGBM.run()
R2_h.append(R2)
MSE_h.append(MSE)
MAE_h.append(MAE)

R2, MSE, MAE = CatBoost.run()
R2_h.append(R2)
MSE_h.append(MSE)
MAE_h.append(MAE)

# R2, MSE, MAE = Neural_Networks.run()
# R2_h.append(R2)
# MSE_h.append(MSE)
# MAE_h.append(MAE)

R2, MSE, MAE = KNN.run()
R2_h.append(R2)
MSE_h.append(MSE)
MAE_h.append(MAE)

R2, MSE, MAE = Bayesian_Linear_Regression.run()
R2_h.append(R2)
MSE_h.append(MSE)
MAE_h.append(MAE)


x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# color = ['r','b','g','y','m']
label = ['LR', 'RR', 'Lasso', 'PR', 'DT', 'RF', 'XGBoost', 'LightGBM', 'CatBoost', 'KNN', 'Bayesion']
plt.bar(x, R2_h, tick_label=label, width=0.5)
plt.tick_params(axis='x', rotation=50)
plt.xlabel('Regression Type')
plt.ylabel('Value')
plt.title('R2')
plt.tight_layout()
# plt.show()
plt.savefig('R2.png')
plt.close

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# color = ['r','b','g','y','m']
label = ['LR', 'RR', 'Lasso', 'PR', 'DT', 'RF', 'XGBoost', 'LightGBM', 'CatBoost', 'KNN', 'Bayesion']
plt.bar(x, MAE_h, tick_label=label, width=0.5)
plt.tick_params(axis='x', rotation=50)
plt.xlabel('Regression Type')
plt.ylabel('Value')
plt.title('MSE')
plt.tight_layout()
# plt.show()
plt.savefig('MSE.png')
plt.close

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# color = ['r','b','g','y','m']
label = ['LR', 'RR', 'Lasso', 'PR', 'DT', 'RF', 'XGBoost', 'LightGBM', 'CatBoost', 'KNN', 'Bayesion']
plt.bar(x, MAE_h, tick_label=label, width=0.5)
plt.tick_params(axis='x', rotation=50)
plt.xlabel('Regression Type')
plt.ylabel('Value')
plt.title('MAE')
plt.tight_layout()
# plt.show()
plt.savefig('MAE.png')
plt.close