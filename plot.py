import numpy as np
import matplotlib.pyplot as plt

N = 200
model_1_output = [np.random.rand(N), np.random.rand(N)]
model_2_output = [np.random.rand(N), np.random.rand(N)]
model_3_output = [np.random.rand(N), np.random.rand(N)]

index_point = [1]
plt.scatter(model_1_output[0][index_point],
            model_1_output[1][index_point], marker='$1$')
plt.scatter(model_2_output[0][index_point],
            model_2_output[1][index_point], marker='$2$')
plt.scatter(model_3_output[0][index_point],
            model_3_output[1][index_point], marker='$3$')
plt.show()
