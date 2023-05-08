import numpy as np
from matplotlib import pyplot as plt
import math
from bezier import curve
import distancemap



if __name__ == '__main__':

    # Define the control points of the cubic curve
    control_points = np.array([(110, 151), (139,152),(161,133),(161,113),(161,113),(162,92),(140,75),(110,73),(110,73),(88,75),(60,91),(60,116),(60,116),(60,130),(82,152),(110, 151)], dtype=np.float32)
    #plot the control points
    plt.scatter(control_points[:,0], control_points[:,1])
    plt.show()



    # using bezier library curved Polygon module plot control points
    from bezier import curved_polygon, curve
   #split the control points into 4 points each.
    control_points = np.split(control_points, len(control_points)/4)
    #turn each set of curves into fortran array as required by bezier library e.g nodes0 = np.asfortranarray([
#   [0.0,  1.0, 2.0,2.0],
#   [0.0, -1.0, 0.0,1.0],
#)



    #now create the curves e.g edge0 = bezier.Curve(control_points[0], degree=3)
    curves = [curve.Curve(i.T, degree=3) for i in control_points]
    #plot the curves on a single image

    # for i in curves:
    #     i.plot(256)
    # plt.show()

    curvedPoly=curved_polygon.CurvedPolygon(curves[0], curves[1],curves[2],curves[3])
    curvedPoly.plot(100)
    plt.show()
    #for each curve in curves evaluate 100 points along each curve and add them to ain ordered list
    n_points = 100
    points_array = np.empty((len(curves)*n_points, 2), dtype=np.float64)
    for i in curves:
        points_array = np.vstack((points_array, i.evaluate_multi(np.linspace(0,1,n_points)).T))
    #change any nan values to 0
    points_array = np.nan_to_num(points_array)

    image=np.zeros((224,224))
    #add the points to the image with a value of 1 at each point
    for i in points_array:
        image[int(i[0]),int(i[1])]=1
    #plot the image
    plt.imshow(image)
    plt.show()
    #plot the points
    #plt.scatter(points[0], points[1])
    plt.show()
    #now create a distance map and plot it
    distanceMap=distancemap.distance_map_from_binary_matrix(image)
    plt.imshow(distanceMap)
    plt.show()


