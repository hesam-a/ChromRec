import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_globular_chain(num_points, step_size, max_attempts=1000, backtrack_steps=5):
    points = []
    points.append(np.array([0.0, 0.0, 0.0]))  # Starting point

    i = 1
    while i < num_points:
        found_valid = False
        attempts = 0
        while not found_valid and attempts < max_attempts:
            attempts += 1
            # Random step in 3D
            step = np.random.normal(size=3)
            step /= np.linalg.norm(step)  # Normalize to unit length
            step *= step_size
            
            new_point = points[-1] + step
            if all(np.linalg.norm(new_point - p) >= step_size for p in points):
                found_valid = True
                points.append(new_point)
        
        if found_valid:
            # Apply globular constraint by moving towards center of mass
            center_of_mass = np.mean(points, axis=0)
            direction_to_center = center_of_mass - points[-1]
            points[-1] += direction_to_center * 0.05  

            # Re-check the minimum distance constraint after applying the globular constraint
            if any(np.linalg.norm(points[-1] - p) < step_size for p in points[:-1]):
                points.pop()
                continue

            i += 1
            print(f"Generated point {i}/{num_points}")
        else:
            # Backtrack if too many attempts have been made
            print(f"Backtracking from point {i} after {attempts} attempts")
            for _ in range(min(backtrack_steps, len(points)-1)):
                points.pop()
            i -= backtrack_steps

    return np.array(points)

# Parameters
num_points = 1000
step_size = 230.0

# Generate chain
chain = generate_globular_chain(num_points, step_size)

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(chain[:, 0], chain[:, 1], chain[:, 2], marker='o', linestyle='-', color='b', markerfacecolor='r')
#ax.plot(chain[:, 0], chain[:, 1], chain[:, 2], marker='o', linestyle='-', color='red', markerfacecolor='yellow', markeredgecolor='black')
#ax.plot(chain[:, 0], chain[:, 1], chain[:, 2], marker='o', linestyle='-', color='darkblue', markerfacecolor='white', markeredgecolor='orange')
#ax.plot(chain[:, 0], chain[:, 1], chain[:, 2], marker='o', linestyle='-', color='green', markerfacecolor='pink', markeredgecolor='purple')
#ax.plot(chain[:, 0], chain[:, 1], chain[:, 2], marker='o', linestyle='-', color='orange', markerfacecolor='lightblue', markeredgecolor='navy')
#ax.plot(chain[:, 0], chain[:, 1], chain[:, 2], marker='o', linestyle='-', color='purple', markerfacecolor='lime', markeredgecolor='red')
plt.show()
