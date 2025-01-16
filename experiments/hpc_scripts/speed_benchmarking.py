import numpy as np
import matplotlib.pyplot as plt

run_time_eta_01 = np.array([15.2,35.1,64.1,149.6,162.1,192.4,282.2,591.2])
num_data_points_eta_01 = np.array([5,10,15,20,25,30,35,50])

run_time_eta_1 = np.array([15.7,19.1,49.2,80.5,93.2,172.3,230.4,232.9,205.9, 380.6,267.3])
num_data_points_eta_1 = np.array([5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000])/1000

a_01,b_01,c_01 = np.polyfit(num_data_points_eta_01,run_time_eta_01,2)
a_1,b_1,c_1 = np.polyfit(num_data_points_eta_1,run_time_eta_1,2)

print(f"{a_1:.2f}x^2 + {b_1:.2f}x + {c_1:.2f}")

x = np.linspace(0,55,1000)
y = [a_1*val**2 + b_1*val + c_1 for val in x]
plt.scatter(num_data_points_eta_1,run_time_eta_1)
plt.plot(x,y)
plt.show()

def predict_time(num_data_points,a,b,c):
    num_data_points/= 1000
    return a*num_data_points**2 + b*num_data_points + c

num_data_points = 100000
print(f"{predict_time(num_data_points,a_1,b_1,c_1)/60} minutes")