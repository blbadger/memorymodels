import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import ScalarFormatter

# Sample data
categories = ['Transformer h=8',
              'Mixer d=1024, k=16',
              'Mixer d=1024, k=8', 
              'Mixer h=4', 
              'Mixer d=512, k=8', 
            #   'Mixer h=4 FineMath', 
              'Transformer h=4',
              'Transformer l=4',
              'Transformer l=8',
              'Mixer k=8',
              'Mixer k=4']
model_architecture = ['Mixer', 
                      'Mixer', 
                      'Mixer', 
                      'Mixer', 
                    #   'Mixer', 
                      'Transformer', 
                      'Transformer', 
                      'Transformer', 
                      'Mixer', 
                      'Mixer']
in_dist = [0.301, 
           1.392, 
           1.493, 
           3.23, 
        #    0.7, 
           2.03, 
           0.291, 
           0.098, 
           0.099, 
           0.148]
random = [47.26, 
          38.303, 
          55.377, 
          18.44, 
        #   13.368, 
          17.94, 
          11.08, 
          10.296, 
          67.60, 
          53.07]
out_dist = [1.36, 
            2.68, 
            2.46, 
            3.56, 
            # 2.54, 
            3.33, 
            1.72, 
            1.07, 
            0.59, 
            0.94]

categories = categories + categories
loss = in_dist + out_dist
distribution = ['In Distribution' for _ in in_dist] + ['Marginally OOD' for _ in random]

data = {
    'categories': categories,
    'data': distribution,
    'loss': loss,
}

model_types = ['Autoencoder' for i in range(5)] + ['Memory Model' for i in range(4)]
data = {
    'In Distribution': in_dist,
    'Marginally OOD': out_dist,
    'Type': model_types,
    'Architecture': model_architecture
}
dataframe = pd.DataFrame(data)
print (dataframe)
sns.set_theme(style="darkgrid")
# Create the bar plot
# g = sns.catplot(data = dataframe, kind='bar', x='loss', y='categories', hue='data', palette='dark', alpha=0.6, height=6)
g = sns.scatterplot(data=dataframe, x='In Distribution', y='Marginally OOD', hue='Architecture')
g = sns.regplot(data=dataframe, x='In Distribution', y='Marginally OOD', scatter=False, logx=True)
# Add labels and title
# plt.legend(fontsize='x-large')
plt.xscale('log')
# for axis in [g.xaxis, g.yaxis]:
#     formatter = ScalarFormatter()
#     formatter.set_scientific(True)
#     axis.set_minor_formatter(formatter)

g.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
# g.get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
# g.grid(which='major', color='w', linewidth=1.0)
# g.grid(which='minor', color='w', linewidth=0.5)

plt.xlabel('In-Distribution Loss', fontsize='large')
plt.ylabel('Marginally OOD Loss', fontsize='large')
plt.ylim(0, 4)
# plt.title('Basic Bar Plot')


plt.tight_layout()

# Display the plot
plt.show()