import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
M=np.genfromtxt('Md.csv', delimiter=',') 
x = np.around(M, 8)
plt.figure(figsize=(16, 14))
s=sns.heatmap(x, annot=True, fmt='g')
s.set_xlabel('True', fontsize=16)
s.set_ylabel('Predict', fontsize=16)
s.set_title('Hospital D')
# s.set(xlabel='Predict Label', ylabel='True Label')