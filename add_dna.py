import numpy as np


def pointGen(P1, P2, npoint, dist):

    # Calculate the direction vector from P1 to P2
    direction_vector = P2 - P1
    
    # Normalize the direction vector to get the unit vector
    norm = np.linalg.norm(direction_vector)
    unit_vector = direction_vector / norm
    
    # Generate and print 5 more points in the same direction, 50 units apart
    points = [P2 + unit_vector * dist * i for i in range(1, npoint+1)]
    for i, point in enumerate(points, start=1):
        print(f"Point {i+2}: {point}")
 
H31 = np.array([131.212189, 126.855835, 137.335938])    # A
H41 = np.array([129.095459, 116.979622, 136.993576])    # B
H32 = np.array([107.114456, 133.206940, 110.713539])    # E
H42 = np.array([112.637611, 127.008202, 104.648575])    # F

# initial points
P1 = np.array([70.801989, 164.146089, 135.811757])
P2 = np.array([77.539348, 127.626739, 113.566544])

P3 = np.array([154.885864, 151.744961, 137.07047])
P4 = np.array([138.634265, 186.337816, 135.96644])

#pointGen(P2, P1, 5, 50)
#print()
#pointGen(P3, P4, 5, 50)

pointGen(H41, H31, 1, 23.5)
print()
pointGen(H42, H32, 1, 23.5)
