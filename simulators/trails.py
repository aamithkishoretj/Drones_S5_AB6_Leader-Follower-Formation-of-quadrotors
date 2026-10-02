"""Bounded, distance-sampled actual-flight trails shared by graphical backends."""
from collections import deque
import numpy as np

COLORS = ((1., .15, .05, 1.), (.05, .45, 1., 1.))


class FlightTrails:
    def __init__(self, max_points=800, spacing=.025):
        self.points = [deque(maxlen=max_points), deque(maxlen=max_points)]
        self.spacing = spacing

    def update(self, positions):
        changed = []
        for i, position in enumerate(positions):
            p = np.asarray(position, float).copy()
            if not self.points[i] or np.linalg.norm(p-self.points[i][-1]) >= self.spacing:
                self.points[i].append(p)
                changed.append(i)
        return changed

    def marker(self, i):
        r, g, b, a = COLORS[i]
        color = f"r:{r} g:{g} b:{b} a:{a}"
        points = " ".join(f"point {{x:{p[0]:.6f} y:{p[1]:.6f} z:{p[2]:.6f}}}" for p in self.points[i])
        return (f'action:ADD_MODIFY ns:"formation_trails" id:{i+1} type:LINE_STRIP '
                'visibility:GUI pose {orientation {w:1}} scale {x:1 y:1 z:1} '
                + 'material {ambient {' + color + '} diffuse {' + color
                + '} emissive {' + color + '}} ' + points)
