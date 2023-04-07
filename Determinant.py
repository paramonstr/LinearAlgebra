# import matplotlib.patches as patches
from LinearTransformation import *


def matrix_to_xy_coordinates(matrix):
    origin = np.array([0, 0])
    ex, ey = matrix_to_vectors(matrix)
    corner = ex + ey
    return [origin, ex, corner, ey]


def plot_matrix(original, transformation):
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
    xmin, xmax, ymin, ymax = -5, 5, -5, 5
    ticks_frequency = 1
    x_ticks = np.arange(xmin, xmax + 1, ticks_frequency)
    y_ticks = np.arange(ymin, ymax + 1, ticks_frequency)

    for ax in axes:
        ax.set(xlim=(xmin - 1, xmax + 1), ylim=(ymin - 1, ymax + 1), aspect='equal')  # create even coordiantee system
        ax.spines['bottom'].set_position('zero')  # draws x axis y = 0
        ax.spines['left'].set_position('zero')  # draws y axis on x = 0
        ax.spines['top'].set_visible(False)  # remove plot borders
        ax.spines['right'].set_visible(False)  # remove plot borders
        ax.set_xlabel('x', size=14, labelpad=-24, x=1.03)
        ax.set_ylabel('y', size=14, labelpad=-21, y=1.02, rotation=0)
        ax.set_xticks(x_ticks[x_ticks != 0])  # remove 0 label on x axis
        ax.set_yticks(y_ticks[y_ticks != 0])  # remove 0 label on x axis
        ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

        # draw original determinant
        axes[0].add_patch(patches.Polygon(xy=matrix_to_xy_coordinates(original), color="green", alpha=0.1))

        # draw transformed determinant
        axes[1].add_patch(patches.Polygon(xy=matrix_to_xy_coordinates(transformation), color="blue", alpha=0.1))

        # titles
        axes[0].set_title(f"original det({original}) = {np.linalg.det(original)}",
                          y=1.08)  # y = 1.08 elevates the title from the plot by 8 %
        axes[1].set_title(f"transformed det({transformation}) = {np.linalg.det(transformation)}",
                          y=1.08)  # y = 1.08 elevates the title from the plot by 8 %

    fig.tight_layout()
    return plt

if __name__ == "__main__":
    original = vectors_to_matrix(np.array([1, 0]), np.array([0, 1]))
    transformation_matrix = vectors_to_matrix(np.array([1, -2]), np.array([3, 0]))

    plot_matrix(original, transformation_matrix).show()
