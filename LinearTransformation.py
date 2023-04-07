import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def matrix_to_xy_coordinates(matrix):
    origin = np.array([0, 0])
    ex, ey = matrix_to_vectors(matrix)
    corner = ex + ey
    return [origin, ex, corner, ey]


def vectors_to_matrix(*arrays):
    """
    :return: vertical transformed and horizontally stacked arrays
    """
    vectors = [np.vstack(array) for array in arrays]
    matrix = np.hstack(vectors)
    return matrix


def matrix_to_vectors(matrix):
    """
    :return: return columns of matrix as vectors, for 2 and 3 dimensions
    """
    # 2d matrix
    if matrix.shape[1] == 2:
        x, y = matrix[:, 0], matrix[:, 1]
        return x, y
    elif matrix.shape[1] == 3:
        x, y, z = matrix[:, 0], matrix[:, 1], matrix[:, 2]
        return x, y, z


def linear_transformation(transformation_matrix, v):
    """
    :param transformation_matrix: matrix of transformed unit vector
    :param v: vector v to the normal basis unit vectors
    :return: transformed vector v to the transformation matrix
    """
    transformed_v = np.matmul(transformation_matrix, np.vstack(v))
    return transformed_v


def plot_2d_transformation(original_vector, transformation_matrix, draw_span=True):
    """
    :param original_vector: original vector to the standard base ex, ey
    :param transformation_matrix: matrix of transformed unit vector
    :param draw_span: default true, will draw the span (determinant) of the original and transformed base
    :return:
    """
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
    system_size = transformation_matrix.max() + 3
    xmin, xmax, ymin, ymax = -system_size, system_size, -system_size, system_size
    ticks_frequency = 1
    x_ticks = np.arange(xmin, xmax + 1, ticks_frequency)
    y_ticks = np.arange(ymin, ymax + 1, ticks_frequency)

    # plot layout
    for ax in axes:
        ax.set(xlim=(xmin - 1, xmax + 1),
               ylim=(ymin - 1, ymax + 1),
               aspect='equal',  # create even coordinate system
               xticks=x_ticks[x_ticks != 0],  # draw ticks and remove 0 label on x axis
               yticks=y_ticks[y_ticks != 0]  # draw ticks and remove 0 label on y axis
               )

        for spine in ["bottom", "left"]:  # draw coordinate center at x = 0 and y = 0 x axis y = 0
            ax.spines[spine].set_position('zero')

        for spine in ["top", "right"]:  # remove plot borders
            ax.spines[spine].set_visible(False)  # remove plot borders

        ax.set_xlabel('x', size=14, labelpad=-24, x=1.03)
        ax.set_ylabel('y', size=14, labelpad=-21, y=1.02, rotation=0)
        ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

    # draw to the original basis and vector
    ex = np.array([1, 0])
    ey = np.array([0, 1])
    original_base = vectors_to_matrix(ex, ey)
    original_vectors = [ex, ey, original_vector]
    for vector in original_vectors:
        if np.array_equal(vector, original_vector):
            color = "red"
        else:
            color = "gray"
        axes[0].quiver(0, 0, vector[0], vector[1], color=color, angles='xy', scale_units='xy', scale=1)

    # draw to the transformed basis and vector
    v_trans = linear_transformation(transformation_matrix, original_vector)
    ex_trans, ey_trans = matrix_to_vectors(transformation_matrix)
    transformed_vectors = [ex_trans, ey_trans, v_trans]
    for vector in transformed_vectors:
        if np.array_equal(vector, v_trans):
            color = "red"
        else:
            color = "gray"
        axes[1].quiver(0, 0, vector[0], vector[1], color=color, angles='xy', scale_units='xy', scale=1)

    # draw span/determinant of original and transformed basis
    if draw_span:
        axes[0].add_patch(patches.Polygon(xy=matrix_to_xy_coordinates(original_base), color="green", alpha=0.1))
        axes[1].add_patch(patches.Polygon(xy=matrix_to_xy_coordinates(transformation_matrix), color="blue", alpha=0.1))

    # titles
    axes[0].set_title(f"original to ex = {ex} and ex = {ey}",
                      y=1.08)  # y = 1.08 elevates the title from the plot by 8 %
    axes[1].set_title(f"transformed on ex = {ex_trans} and ex = {ey_trans}",
                      y=1.08)  # y = 1.08 elevates the title from the plot by 8 %

    fig.tight_layout()
    return plt


if __name__ == "__main__":
    v = np.array([10, 2])
    transformation_matrix = vectors_to_matrix(np.array([-3, 1]), np.array([-4, 0]))
    plot_2d_transformation(v, transformation_matrix).show()
