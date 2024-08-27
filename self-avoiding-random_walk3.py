import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_globular_chain(num_points, step_size, max_attempts=1000):
    points = []
    points.append(np.array([0.0, 0.0, 0.0]))  # Starting point

    for i in range(1, num_points):
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            # Random step in 3D
            step = np.random.normal(size=3)
            step /= np.linalg.norm(step)  # Normalize to unit length
            step *= step_size

            new_point = points[-1] + step

            # Check for non-overlapping constraint
            if all(np.linalg.norm(new_point - p) >= step_size for p in points):
                points.append(new_point)
                break

        if attempts >= max_attempts:
            print(f"Failed to place point {i}. Backtracking to maintain continuity.")
            # Backtrack a few steps if too many attempts have been made
            for _ in range(5):
                if points:
                    points.pop()
            i -= 5
            i = max(1, i)

    points = np.array(points)

    # Apply energy minimization to create globular structure
    def energy(points):
        center_of_mass = np.mean(points, axis=0)
        return np.sum(np.linalg.norm(points - center_of_mass, axis=1))

    def monte_carlo_optimization(points, num_iterations=10000, temperature=1.0):
        for _ in range(num_iterations):
            # Select a random point to move (except the first one to maintain chain start)
            idx = np.random.randint(1, len(points))

            # Propose a new position for this point
            step = np.random.normal(size=3) * step_size * 0.1
            new_point = points[idx] + step

            # Calculate energy difference
            original_energy = energy(points)
            points[idx] = new_point
            new_energy = energy(points)

            # Accept or reject the new position based on energy change
            if new_energy > original_energy:
                # Metropolis criterion
                if np.exp(-(new_energy - original_energy) / temperature) < np.random.rand():
                    points[idx] = points[idx] - step  # Reject the move

            # Gradually cool down the temperature
            temperature *= 0.999

    monte_carlo_optimization(points)
    
    return points

# Parameters
num_points = 1000
step_size = 230.0

# Generate chain
chain = generate_globular_chain(num_points, step_size)

# Plot the chain
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(chain[:,0], chain[:,1], chain[:,2])
plt.show()
