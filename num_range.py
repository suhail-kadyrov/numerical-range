import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError


def draw_numerical_range(matrix, points_count=5_000_000, boundary=True, interior=False, eigenvalues=False):
	assert type(matrix) in (list, tuple), 'The parameter `matrix` shuold be of type list or tuple.'
	assert type(points_count) == int and points_count > 0, 'The parameter `points_count` shuold be positive integer.'
	assert boundary or interior or eigenvalues, 'At least one of the parameters `boundary`, `interior`, and `eigenvalues` ​​should be True.'
	matrix_ = np.array(matrix)
	assert matrix_.ndim == 2, 'The parameter `matrix` should be two-dimensional.'
	assert matrix_.dtype in (np.int64, np.float64, np.complex128), 'Elements of the parameter `matrix` should be numbers.'
	assert matrix_.shape[0] == matrix_.shape[1], 'The concept of numerical range is available only for square matrices.'
	n_ = matrix_.shape[0]
	x = np.random.randn(n_, points_count) + 1j * np.random.randn(n_, points_count)
	norm_x = np.linalg.norm(x, axis=0)
	x_ = np.divide(x, norm_x)
	w = np.multiply(np.conj(x_), np.matmul(matrix_, x_)).sum(axis=0)
	eigvals = np.linalg.eigvals(matrix_)
	points = np.vstack((w.real, w.imag)).T
	plt.figure(figsize=(12, 8))
	if boundary:
		try:
			hull = ConvexHull(points)
		except QhullError:
			interior = True
		else:
			for simplex in hull.simplices:
				plt.plot(points[simplex, 0], points[simplex, 1], 'black')
	if interior:
		sns.scatterplot(x=w.real, y=w.imag, label='Interior points')
	if eigenvalues:
		sns.scatterplot(x=eigvals.real, y=eigvals.imag, color='red', label='Eigenvalues')
	plt.title('Numerical range')
	plt.axis('equal')
	plt.xlabel('real')
	plt.ylabel('imaginary')
	plt.grid(True, linestyle='-.')
	if interior or eigenvalues:
		plt.legend(loc=1)
	plt.show()


if __name__ == '__main__':
	draw_numerical_range(
		matrix=(
			(4, 0, 0),
			(0, 2, 3),
			(0, 0, 1),
		)
	)
