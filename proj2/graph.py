import matplotlib.pyplot as plt
import numpy as np

n10 = np.mean([0.000, 0.000, 0.000, 0.000, 0.000])
n100 = np.mean([0.002, 0.002, 0.002, 0.002, 0.002])
n1000 = np.mean([0.015, 0.014, 0.014, 0.013, 0.014])
n10000 = np.mean([0.080, 0.083, 0.081, 0.081, 0.089])
n100000 = np.mean([0.721, 0.685, 0.820, 0.714, 0.722])
n500000 = np.mean([4.639, 4.059, 4.754, 4.073, 4.299])
n1000000 = np.mean([8.332, 8.333, 8.254, 8.276, 8.345])

x = [10, 100, 1000, 10000, 100000, 500000, 1000000]
y = [n10, n100, n1000, n10000, n100000, n500000, n1000000]

coeff = np.polyfit(x, y, 1)

print(coeff)

x_fit = np.linspace(10, 1000000, num=100)
y_fit = coeff[0] * x_fit + coeff[1]

y_log = 0.0000006 * x_fit * np.log(x_fit) + 0.05
y_sq = x_fit * x_fit
y_n = x_fit

plt.scatter(x, y)
plt.plot(x_fit, y_fit, label="polyfit")
plt.plot(x_fit, y_log, label="0.0000006 n log(n) + 0.05")
# plt.plot(x_fit, y_n, label="n")
plt.legend()
plt.xscale("log")
plt.show()
