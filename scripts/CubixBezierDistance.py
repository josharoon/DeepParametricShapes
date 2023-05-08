import numpy as np
from matplotlib import pyplot as plt
import math
from bezier import curve
def closest_point_on_cubic_curve(a, b, c, d, p):
    # Define the coefficients of the quintic equation
    A = -a + 3*b - 3*c + d
    B = 3*a - 6*b + 3*c
    C = -3*a + 3*b
    D = a - p
    
    coeffs = [
        np.dot(A, A),
        2 * np.dot(A, B),
        np.dot(B, B) + 2 * np.dot(A, C),
        2 * np.dot(B, C) + 2 * np.dot(A, D),
        np.dot(C, C) + 2 * np.dot(B, D),
        2 * np.dot(C, D),
        np.dot(D, D)
    ]
    
    # Find the roots of the quintic equation
    roots = np.roots(coeffs)
    
    # Select the root that gives the closest point on the curve to p
    closest_t = roots[np.argmin(np.abs(roots.imag))].real
    
    # Evaluate the curve at the closest t
    closest_point = (1 - closest_t)**3 * a + 3 * closest_t * (1 - closest_t)**2 * b + \
                    3 * closest_t**2 * (1 - closest_t) * c + closest_t**3 * d
    
    return closest_point

def distance_to_union_of_cubic_curves(control_points, p):
    closest_points = []
    for i in range(0, len(control_points), 4):
        a, b, c, d = control_points[i:i+4]
        closest_points.append(closest_point_on_cubic_curve(a, b, c, d, p))
    
    # Compute the distance to the closest point on each curve
    distances = np.linalg.norm(np.array(closest_points) - p, axis=1)
    
    # Compute the minimum distance
    min_distance = np.min(distances)
    
    return min_distance


if __name__ == '__main__':

    # Define the control points of the cubic curve
    control_points = np.array([(110, 151), (139,152),(161,133),(161,113),(161,113),(162,92),(140,75),(110,73),(110,73),(88,75),(60,91),(60,116),(60,116),(60,130),(82,152),(110, 151)], dtype=np.float32)
    #plot the control points
    plt.scatter(control_points[:,0], control_points[:,1])
    plt.show()



#     # using bezier library curved Polygon module plot control points
#     from bezier import curved_polygon, curve
#    #split the control points into 4 points each.
#     control_points = np.split(control_points, len(control_points)/4)
#     #turn each set of curves into fortran array as required by bezier library e.g nodes0 = np.asfortranarray([
# #   [0.0,  1.0, 2.0,2.0],
# #   [0.0, -1.0, 0.0,1.0],
# #)
#     #convert control points to series of 4x2 arrays
#     control_points=[np.reshape(i, (2,4)) for i in control_points]
#
#
#     control_pointsFT = [np.asfortranarray(curve) for curve in control_points]
#     #now create the curves e.g edge0 = bezier.Curve(control_points[0], degree=3)
#     curves = [curve.Curve(i, degree=3) for i in control_points]
#     #plot the curves on a single image
#
#     for i in curves:
#         i.plot(256)
#     plt.show()
#
#     curvedPoly=curved_polygon.CurvedPolygon(curves[0], curves[1],curves[2],curves[3])
#     curvedPoly.plot()
#     plt.show()


    control_points=control_points[:4]
    bezierCurve=curve.Curve(control_points.T, degree=3)
    bezierCurve.plot(256)
    plt.show()
    #create empty 224x224 grid and for each point get distance to union then plot as an image
    grid = np.zeros((224,224))
    for i in range(224):
        for j in range(224):
            closestpoint= closest_point_on_cubic_curve(control_points[0], control_points[1], control_points[2], control_points[3], (i,j))
            #get distance between i,j and closest point
            grid[i,j]=np.linalg.norm(np.array(closestpoint) - np.array((i,j)))
    #plot as a grayscale image on a 224x224 grid
    plt.imshow(grid, cmap='gray')
    plt.show()


