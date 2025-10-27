import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import html
import webbrowser
from transformers import AutoTokenizer
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# attr_dict = json.load(open('loss_exact_calculations.json'))['entropies']
# loss_dict = json.load(open('big_loss_attributions.json'))['losses']
# loss_dict = json.load(open('entropy_estimates/clm_losses.json'))['losses']
# all_losses = []
# all_attr = []
# for key in loss_dict:
#     if key in attr_dict:
#         # print (len(attr_dict[key]))
#         loss = loss_dict[key][511:1023]
#         attr = attr_dict[key][1][512:1024]
#         # print (len(loss), len(attr))
#         all_losses += loss
#         all_attr += attr

# for key in loss_dict:
#     if key in loss_dict2:
#         print (len(loss_dict2[key]))
#         loss = loss_dict[key][1][:-1]
#         attr = loss_dict2[key]
#         print (len(loss), len(attr))
#         all_losses += loss
#         all_attr += attr

loss_dict = json.load(open('entropy_estimates/clm_losses.json'))['losses']
attr_dict = json.load(open('fineweb_2eem_shifted.json'))['losses']

all_losses = []
all_attr = []
for key in loss_dict:
    if key in attr_dict:
        loss = loss_dict[key]
        attr = attr_dict[key][1][1:]
        print (loss, attr)
        all_losses += loss
        all_attr += attr


data = pd.DataFrame({'CLM Loss': all_losses, '2nd order EEM': all_attr})
sns.regplot(x='CLM Loss', y='2nd order EEM', data=data, marker='.', scatter_kws={'s':4, 'alpha':0.6})
# plt.xscale('log')
plt.show()
plt.close()

model = LinearRegression()
model.fit(np.array(all_losses).reshape(-1, 1), np.array(all_attr))

slope = model.coef_[0]
bias = model.intercept_
r2_score_method = model.score(np.array(all_losses).reshape(-1, 1), np.array(all_attr))
y_pred = model.predict(np.array(all_losses).reshape(-1, 1))
r2_score_func = r2_score(np.array(all_attr), y_pred)
print (f"MSE: {mean_squared_error(all_losses, all_attr)}")
print(f"Slope (coefficient): {slope:.4f}")
print(f"Bias (intercept): {bias:.4f}")
print(f"R-squared (from model.score()): {r2_score_method:.4f}")
print(f"R-squared (from sklearn.metrics.r2_score): {r2_score_func:.4f}")


