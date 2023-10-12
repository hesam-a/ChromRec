import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def calculate_direction(point1, point2):
    return (np.array(point2) - np.array(point1)) / calculate_distance(point1, point2)

def calculate_new_site_position(com1, com2, radius, min_overlap_ratio=0.55, max_overlap_ratio=0.65):
    distance = calculate_distance(com1, com2)
    direction = calculate_direction(com1, com2)
    min_overlap = radius - (min_overlap_ratio * radius)
    max_overlap = radius - (max_overlap_ratio * radius)
    chosen_overlap = (min_overlap + max_overlap) / 2
    new_site_position = np.array(com1) + (direction * chosen_overlap)
    return new_site_position

def calculate_LAK_mean(com1, new_site_position):
    return np.linalg.norm(np.array(new_site_position) - np.array(com1))

com1 = np.array([-9, 0, 0])
com2 = np.array([9, 0, 0])
#com2 = np.array([25, 25, 25])
radius1 = 18.489
radius2 = 17.054

site1_pos = calculate_new_site_position(com1, com2, radius1)
site2_pos = calculate_new_site_position(com2, com1, radius2)
distance_1 = calculate_LAK_mean(site1_pos, com1)
distance_2 = calculate_LAK_mean(site2_pos, com2)
site1_site2 = calculate_LAK_mean(site2_pos, site1_pos)
print("site1_site2 distance: ", site1_site2)
print("com1 - com2 distance: ", calculate_LAK_mean(com1, com2))
print("site1 position: ", site1_pos)
print("site2 position: ", site2_pos)

# Place the two centers of mass along the x-axis and 14.5 units apart
com1_adjusted = np.array([0, 0])
com2_adjusted = np.array([14.4, 0])

# Calculate new site positions along the x-axis
site1_pos_adjusted = calculate_new_site_position(com1_adjusted, com2_adjusted, radius1)
site2_pos_adjusted = calculate_new_site_position(com2_adjusted, com1_adjusted, radius2)

distance_1 = calculate_LAK_mean(site1_pos_adjusted, com1_adjusted)
distance_2 = calculate_LAK_mean(site2_pos_adjusted, com2_adjusted)

# Ensure the y-coordinate of the site positions is 0 (along the x-axis)
site1_pos_adjusted[1] = 0
site2_pos_adjusted[1] = 0

fig = plt.figure(figsize=(12, 8), dpi=100)
ax = fig.add_subplot(111)

# Plot centers of mass
ax.scatter(*com1_adjusted, c='r', s=100, label='Center of Mass 1', zorder=5)
ax.scatter(*com2_adjusted, c='b', s=100, label='Center of Mass 2', zorder=5)
ax.text(1, 2,   f'dist of site1 to COM1: {distance_1:.2f} Å', color='red', fontsize=15)
ax.text(10, -2, f'dist of site2 to COM2: {distance_2:.2f} Å', color='blue', fontsize=15)
ax.text(4.5, 20, f'60% overlap', color='blue', fontsize=15)

# Plot line between centers of mass
ax.plot([com1_adjusted[0], com2_adjusted[0]], [com1_adjusted[1], com2_adjusted[1]], '--', color="black", linewidth=2, label='Line between centers')

# Plot new site position
ax.scatter(*site1_pos_adjusted, c='purple', s=100, label='Site 1 Position', zorder=5)
ax.scatter(*site2_pos_adjusted, c='orange', s=100, label='Site 2 Position', zorder=5)

# Draw circles around each center of mass with the given radii
circle1 = plt.Circle(com1_adjusted, radius1, color='#d540ff', alpha=0.2, linewidth=0)
circle2 = plt.Circle(com2_adjusted, radius2, color='#d540ff', alpha=0.2, linewidth=0)
ax.add_patch(circle1)
ax.add_patch(circle2)

# Labels and legend
ax.set_xlabel('X', fontsize=14, fontweight='bold')
ax.set_ylabel('Y', fontsize=14, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(fontsize=14,prop={'weight':'bold'})
plt.xlim(-25,52)
plt.ylim(-25,25)
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)

plt.show()

