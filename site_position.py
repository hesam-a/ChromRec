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
    #print("direction: ", direction)
    min_overlap = radius - (min_overlap_ratio * radius)
    max_overlap = radius - (max_overlap_ratio * radius)
    chosen_overlap = (min_overlap + max_overlap) / 2
    new_site_position = np.array(com1) + (direction * chosen_overlap)
    return new_site_position

def calculate_LAK_mean(com1, new_site_position):
    return np.linalg.norm(np.array(new_site_position) - np.array(com1))

com1 = np.array([-10.5, 0, 0])
com2 = np.array([ 10.5, 0, 0])
#com1 = np.array([-6.0754, 0.0517,  0.0397])
#com2 = np.array([ 6.0263, 0.0176, -0.0207])
radius1 = 18.489
radius2 = 17.054

site1_pos = calculate_new_site_position(com1, com2, radius1)
site2_pos = calculate_new_site_position(com2, com1, radius2)
distance_1 = calculate_LAK_mean(site1_pos, com1)
distance_2 = calculate_LAK_mean(site2_pos, com2)
site1_site2 = calculate_LAK_mean(site2_pos, site1_pos)
print("com1 - com2 distance: ", calculate_LAK_mean(com1, com2))
print("site1_site2 distance: ", round(site1_site2,3))
print("site1 position: ", site1_pos)
print("site2 position: ", site2_pos)


fig = plt.figure(figsize=(12, 8), dpi=100)  # Increase DPI for better resolution
ax = fig.add_subplot(111, projection='3d')

# Plot centers of mass
ax.scatter(*com1, c='r', s=100, label='Center of Mass 1', zorder=5)  # Increased size
ax.scatter(*com2, c='b', s=100, label='Center of Mass 2', zorder=5)  # Increased size

# Plot line between centers of mass
ax.plot([com1[0], com2[0]], [com1[1], com2[1]], [com1[2], com2[2]], '--', color="#0f00cf", linewidth=2, label='Line between centers')
#ax.plot([com2[0], site2_pos[0]], [com2[1], site2_pos[1]], [com2[2], site2_pos[2]], '--', color="#0f00cf", linewidth=2, label='Line between centers')

# Plot new site position
ax.scatter(*site1_pos, c='purple', s=100, label='Site 1 Position', zorder=5)  # Increased size
ax.scatter(*site2_pos, c='orange', s=100, label='Site 2 Position', zorder=5)  # Increased size
#ax.text(0, 0, -5, f'Distance to Center 1: {distance_1:.2f}', color='red', fontsize=15)
#ax.text(21, 3.8, 17, f'Distance to Center 2: {distance_2:.2f}', color='blue', fontsize=15)

## Highlight overlap range
min_overlap1 = radius1 - (0.25 * radius1)
max_overlap1 = radius1 - (0.15 * radius1)
min_overlap2 = radius2 - (0.25 * radius2)
max_overlap2 = radius2 - (0.15 * radius2)
#for center in [com1, com2]:
#    for overlap in [min_overlap, max_overlap]:
#        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]  # Increased resolution
#        x = center[0] + overlap*np.cos(u)*np.sin(v)
#        y = center[1] + overlap*np.sin(u)*np.sin(v)
#        z = center[2] + overlap*np.cos(v)
#        ax.plot_surface(x, y, z, color='#d540ff', alpha=0.2, linewidth=0)

for overlap in [min_overlap1, max_overlap1]:
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]  # Increased resolution
    x = com1[0] + overlap*np.cos(u)*np.sin(v)
    y = com1[1] + overlap*np.sin(u)*np.sin(v)
    z = com1[2] + overlap*np.cos(v)
    ax.plot_surface(x, y, z, color='#d540ff', alpha=0.2, linewidth=0)

for overlap in [min_overlap2, max_overlap2]:
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]  # Increased resolution
    x = com2[0] + overlap*np.cos(u)*np.sin(v)
    y = com2[1] + overlap*np.sin(u)*np.sin(v)
    z = com2[2] + overlap*np.cos(v)
    ax.plot_surface(x, y, z, color='#d540ff', alpha=0.2, linewidth=0)

# Labels and legend
ax.set_xlabel('X', fontsize=14)
ax.set_ylabel('Y', fontsize=14)
ax.set_zlabel('Z', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(fontsize=12)
#ax.grid(True)

plt.show()
