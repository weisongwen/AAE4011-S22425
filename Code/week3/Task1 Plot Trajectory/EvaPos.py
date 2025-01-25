import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# Load the CSV files
# file_path1 = '//wsl.localhost/Ubuntu-20.04/home/wws/ws_VIW-Fusion/src/output/Solution1.csv'  # Replace with your first CSV file path
# file_path2 = '//wsl.localhost/Ubuntu-20.04/home/wws/ws_VIW-Fusion/src/output/Solution2.csv'  # Replace with your second CSV file path

file_path1 = 'C:/Users/Weisong Wen/Dropbox/PolyU AAE Teaching-Weisong/PolyU AAE4011 Artificial Intelligence in Unmanned Autonomous Systems/AAE4011-S22425-2/Code/week3/Task1 Plot Trajectory/Solution1.csv'  # Replace with your first CSV file path
file_path2 = 'C:/Users/Weisong Wen/Dropbox/PolyU AAE Teaching-Weisong/PolyU AAE4011 Artificial Intelligence in Unmanned Autonomous Systems/AAE4011-S22425-2/Code/week3/Task1 Plot Trajectory/Solution2.csv'  # Replace with your second CSV file path

data1 = pd.read_csv(file_path1)
data2 = pd.read_csv(file_path2)
time1 = data1['Time']
data1['Time'] = (data1['Time'] - time1.iloc[0])/(1e09)
pos1 = data1[['PosX', 'PosY', 'PosZ']]
vel1 = data1[['VelX', 'VelY', 'VelZ']]


quat1 = data1[['QuatW', 'QuatX', 'QuatY', 'QuatZ']]

time2 = data2['Time']
data2['Time'] = (data2['Time'] - time2.iloc[0])/(1e09)
# data2['Time'] = data2['Time'] - time2.iloc[0]
pos2 = data2[['PosX', 'PosY', 'PosZ']]
vel2 = data2[['VelX', 'VelY', 'VelZ']]
quat2 = data2[['QuatW', 'QuatX', 'QuatY', 'QuatZ']]

# Convert quaternions to Euler angles (roll, pitch, yaw)
def quat_to_euler(quat):
    r = R.from_quat(quat)
    euler = r.as_euler('xyz', degrees=True)
    return euler

euler1 = quat1.apply(lambda row: quat_to_euler([row['QuatW'], row['QuatX'], row['QuatY'], row['QuatZ']]), axis=1)
euler2 = quat2.apply(lambda row: quat_to_euler([row['QuatW'], row['QuatX'], row['QuatY'], row['QuatZ']]), axis=1)

euler1 = pd.DataFrame(euler1.tolist(), columns=['Roll', 'Pitch', 'Yaw'])
euler2 = pd.DataFrame(euler2.tolist(), columns=['Roll', 'Pitch', 'Yaw'])


# Create a figure with four subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# First subplot
axs[0, 0].plot(pos1['PosX'], pos1['PosY'], marker='', linestyle='-', color='r',label='Baseline')
# axs[0, 0].plot(pos2['PosX'], pos2['PosY'], marker='', linestyle='-', color='b',label='Proposed')
axs[0, 0].set_title('Trajectory (X-Y)')
axs[0, 0].set_xlabel('X direction')
axs[0, 0].set_ylabel('Y direction')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Second subplot
axs[0, 1].plot(data1['Time'], pos1['PosX'], marker='', linestyle='-', color='r',label='Baseline')
axs[0, 1].plot(data2['Time'], pos2['PosX'], marker='', linestyle='-', color='b',label='Proposed')
axs[0, 1].set_title('X direction')
axs[0, 1].set_xlabel('time')
axs[0, 1].set_ylabel('X')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Third subplot
axs[1, 0].plot(data1['Time'], pos1['PosY'], marker='', linestyle='-', color='r',label='Baseline')
axs[1, 0].plot(data2['Time'], pos2['PosY'], marker='', linestyle='-', color='b',label='Proposed')
axs[1, 0].set_title('Y direction')
axs[1, 0].set_xlabel('time')
axs[1, 0].set_ylabel('Y')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Fourth subplot
axs[1, 1].plot(data1['Time'], pos1['PosZ'], marker='', linestyle='-', color='r',label='Baseline')
axs[1, 1].plot(data2['Time'], pos2['PosZ'], marker='', linestyle='-', color='b',label='Proposed')
axs[1, 1].set_title('Z direction')
axs[1, 1].set_xlabel('time')
axs[1, 1].set_ylabel('Z')
axs[1, 1].legend()
axs[1, 1].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot as an image file (optional)
plt.savefig('four_subplots.png')

# Show the plot
plt.show()




