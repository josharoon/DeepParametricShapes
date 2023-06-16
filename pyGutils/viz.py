import bezier
import numpy as np
from matplotlib import pyplot as plt
import torch

#plot output to a tensoboard writer


def plot_to_tensorboard(writer, epoch, output,name="output"):
    # plot output array as x,y points using matplotlib with index values next to each point
    x=output[:,0]
    y=output[:,1]
    plt.plot(x, y, 'ro')
    for i, txt in enumerate(range(len(x))):
        plt.annotate(txt, (x[i], y[i]))
    writer.add_figure(name, plt.gcf(), epoch)

def image_to_tensorboard(writer, epoch, image,name="image"):
    #if image is float tensor, convert to uint8
    if image.dtype == torch.float32:
        image = image.mul(255).byte()
    if len(image.shape) == 2:
        image = image.unsqueeze(0)

    writer.add_image(name, image, epoch)




def plot_distance_field(distance_field, vmax, title='Distance Field',ax=None):
    if ax is None:
        plt.figure(figsize=(6, 6))
    else:
        plt.sca(ax)

    # make sure df is on cpu
    distance_field = distance_field.cpu()
    plt.imshow(distance_field, extent=(0, 1, 0, 1), origin='lower', cmap='viridis', vmin=0, vmax=vmax)
    plt.colorbar(label='distance Field')
    plt.title(title)
    if ax is None:
        plt.show()

def plotCubicSpline(control_points, title='Cubic Bezier Spline from Control Points', ax=None, image=None):
    if ax is None:
        plt.figure(figsize=(6, 6))
    else:
        plt.sca(ax)

    if(image is not None):
        plt.imshow(image, extent=[0, 1, 0, 1])

    #make sure control points are on cpu
    control_points = control_points.cpu()
    for curve in control_points:
        nodes = np.asfortranarray(curve.numpy().T)
        curve = bezier.Curve(nodes, degree=3)
        curve.plot(num_pts=256, ax=plt.gca())
        #
        plt.plot(curve.nodes[0, :2], curve.nodes[1, :2], linestyle='--', marker='o', color='gray')
        plt.plot(curve.nodes[0, 1:], curve.nodes[1, 1:], linestyle='--', marker='o', color='gray')
    #
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    if ax is None:
        plt.show()

def plotQuadraticSpline(control_points, title='Quadratic Bezier Spline from Control Points', ax=None):
    if ax is None:
        plt.figure(figsize=(6, 6))
    else:
        plt.sca(ax)

    # Make sure control points are on CPU


    # Iterate over each shape in the tensor
    for shape in control_points:
        shape = shape.cpu()
    #for curve in control_points:
        for curve in shape:
            nodes = np.asfortranarray(curve.numpy().T)
            curve = bezier.Curve(nodes, degree=2)
            curve.plot(num_pts=256, ax=plt.gca())
            #
            plt.plot(curve.nodes[0, :2], curve.nodes[1, :2], linestyle='--', marker='o', color='gray')
            plt.plot(curve.nodes[0, 1:], curve.nodes[1, 1:], linestyle='--', marker='o', color='gray')
    #
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    if ax is None:
        plt.show()

def plotQuadraticSplineGrid(control_points, title='Quadratic Bezier Spline from Control Points', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Iterate over each shape in the tensor
    for shape in control_points:
        shape = shape.cpu()
        for curve in shape:
            nodes = np.asfortranarray(curve.numpy().T)
            curve = bezier.Curve(nodes, degree=2)
            curve.plot(num_pts=256, ax=ax)
            #
            ax.plot(curve.nodes[0, :2], curve.nodes[1, :2], linestyle='--', marker='o', color='gray')
            ax.plot(curve.nodes[0, 1:], curve.nodes[1, 1:], linestyle='--', marker='o', color='gray')
    #
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)



from utils import compute_alignment_fields

def visualize_vector_field(alignment_fields,diffField=False,scale=1,subsample=5):

    if diffField:
        alignment_fields = compute_alignment_fields(alignment_fields)
    # convert the PyTorch tensor to numpy array
    alignment_fields_np = alignment_fields.cpu().numpy()

    # assuming alignment_fields_np is of shape [H, W, 2], where H is the height, W is the width
    # and 2 for the 2D vector (dx, dy)
    dx = alignment_fields_np[:,:,0]
    dy = alignment_fields_np[:,:,1]


    # Normalize the vectors
    magnitude = np.sqrt(dx**2 + dy**2)
    dx /= magnitude
    dy /= magnitude

    # make a grid
    Y, X = np.mgrid[0:dx.shape[0], 0:dx.shape[1]]

    # Subsample grid and vectors (take every nth point in each direction)
    X = X[::subsample, ::subsample]
    Y = Y[::subsample, ::subsample]
    dx = dx[::subsample, ::subsample]
    dy = dy[::subsample, ::subsample]

    # plot quiver
    plt.quiver(X, Y, dx, dy, color='r')

    plt.show()





if __name__ == '__main__':
    import torch

    control_pointsA = [0.17, 0.9, 0.23, 0.9, 0.28, 0.9, 0.32, 0.77, 0.36, 0.65, 0.42, 0.65, 0.5, 0.65, 0.58, 0.65, 0.65,
        0.65, 0.68, 0.76, 0.73, 0.9, 0.78, 0.9, 0.84, 0.9, 0.81, 0.79, 0.76, 0.67, 0.74, 0.59, 0.7, 0.48, 0.66,
        0.36, 0.63, 0.27, 0.6, 0.2, 0.57, 0.1, 0.52, 0.1, 0.44, 0.1, 0.42, 0.17, 0.39, 0.27, 0.36, 0.34, 0.33,
        0.43, 0.3, 0.52, 0.27, 0.6, 0.24, 0.71, 0.48, 0.29, 0.43, 0.42, 0.38, 0.56, 0.5, 0.57, 0.62, 0.57, 0.58,
        0.45, 0.54, 0.32, 0.5, 0.19] + [0.5]*16

