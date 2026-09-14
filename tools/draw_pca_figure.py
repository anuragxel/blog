"""Draw PCA's projection/reconstruction geometry with Matplotlib.

    python3 tools/draw_pca_figure.py

Requires NumPy and Matplotlib. Saves a PNG only. The centered point cloud and
its principal direction are computed, so the perpendiculars and decomposition
shown in the figure are actual orthogonal projections, not freehand geometry.
"""

import os
from pathlib import Path
from tempfile import TemporaryDirectory


def draw():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    blue, green, ink = '#3538b5', '#17803d', '#252525'
    plt.rcParams.update({'font.family': ['Comic Sans MS', 'DejaVu Sans'],
                         'font.size': 13, 'text.color': ink,
                         'mathtext.fontset': 'dejavusans'})

    # Paired points give an exactly centered, elongated cloud.
    along = np.array([.25, .65, 1.05, 1.45, 1.8, 2.15, 2.5])
    across = np.array([.65, -.48, .34, -.42, .52, -.18, .10])
    theta = np.deg2rad(24)
    basis = np.array([[np.cos(theta), np.sin(theta)],
                      [-np.sin(theta), np.cos(theta)]])
    half = np.column_stack((along, across)) @ basis
    X = np.vstack((half, -half))
    _, eigenvectors = np.linalg.eigh(X.T @ X)
    p = eigenvectors[:, -1]
    if p[0] < 0:
        p = -p
    normal = np.array([-p[1], p[0]])
    scores = X @ p
    reconstructed = scores[:, None] * p
    residuals = X - reconstructed
    assert np.allclose(residuals @ p, 0)
    assert np.isclose(np.sum(X**2), np.sum(scores**2)+np.sum(residuals**2))

    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
    fig.subplots_adjust(left=.025, right=.975, bottom=.19, top=.98)
    ax.set_aspect('equal')
    ax.set_xlim(-3.25, 3.35)
    ax.set_ylim(-1.95, 2.0)
    ax.axis('off')
    ax.annotate('', xy=3.03*p, xytext=-3.03*p,
                arrowprops={'arrowstyle': '->', 'lw': 1.6, 'color': ink})
    ax.text(*(3.03*p+np.array([.07, .06])), '$p$', fontsize=16)
    for x, projected in zip(X, reconstructed):
        ax.plot([x[0], projected[0]], [x[1], projected[1]],
                color=green, lw=1.25, linestyle=(0,(3,3)), zorder=1)
    ax.scatter(*X.T, s=22, facecolors='white', edgecolors=blue, linewidths=1.5, zorder=3)
    ax.scatter(*reconstructed.T, s=12, color=green, zorder=2)

    index = 4
    x, projected = X[index], reconstructed[index]
    ax.text(*(x+np.array([.09,.13])), '$x$', color=blue, fontsize=15)
    ax.text(*(projected+np.array([.10,-.30])), r'$\hat{x}$', color=green, fontsize=15)
    ax.annotate('perpendicular error', xy=(x+projected)/2,
                xytext=(-.65, 1.78), ha='center', color=green,
                arrowprops={'arrowstyle':'-', 'color':green, 'lw':1.1,
                            'connectionstyle':'arc3,rad=-.15'})

    # A parallel span marks variation retained along the projection direction.
    span_a = -2.4*p - 1.0*normal
    span_b = 2.4*p - 1.0*normal
    ax.annotate('', xy=span_b, xytext=span_a,
                arrowprops={'arrowstyle':'<->', 'color':blue, 'lw':1.4})
    ax.text(*(-1.33*normal), 'spread along p', color=blue, ha='center', va='top')
    fig.text(.5, .062, r'$\|X\|_F^2 = \|Xp\|^2 + \|X-(Xp)p^T\|_F^2$',
             ha='center', fontsize=14)

    destination = Path(__file__).resolve().parents[1] / 'assets/images/pca'
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination/'projection-reconstruction.png', facecolor='white', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    with TemporaryDirectory(prefix='blog-matplotlib-') as cache:
        os.environ.setdefault('MPLCONFIGDIR', cache)
        draw()
