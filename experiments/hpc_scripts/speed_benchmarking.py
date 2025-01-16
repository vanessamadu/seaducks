import numpy as np
import matplotlib.pyplot as plt

run_time_eta_01 = np.array([15.2,35.1,64.1,149.6,162.1,192.4,282.2,591.2])
num_data_points_eta_01 = np.array([5,10,15,20,25,30,35,50])

a,b,c = np.polyfit(num_data_points_eta_01,run_time_eta_01,2)

print(f"{a:.2f}x^2 + {b:.2f}x + {c:.2f}")

x = np.linspace(0,50,1000)
y = [a*val**2 + b*val + c for val in x]
plt.scatter(num_data_points_eta_01,run_time_eta_01)
plt.plot(x,y)
plt.show()

def predict_time(num_data_points,a,b,c):
    num_data_points/= 1000
    return a*num_data_points**2 + b*num_data_points + c

num_data_points = 100000
print(f"{predict_time(num_data_points,a,b,c)/60} minutes")