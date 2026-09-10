"""Project accumulated scan regions without Boolean polygon unions."""
from __future__ import annotations

import math

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainterPath


def project_regions(regions, project) -> QPainterPath:
    """Union scans by winding count, preserving the holes in each scan.

    Qt Boolean unions can invert coverage at nearly coincident scan edges.
    Consistent outer/hole winding lets the painter accumulate coverage directly:
    overlapping scans add, and a hole only cancels its own enclosing scan.
    """
    path = QPainterPath()
    path.setFillRule(Qt.FillRule.WindingFill)
    for region in regions:
        for index, ring in enumerate([region.get("outer", [])] + list(
            region.get("holes", [])
        )):
            if len(ring) < 3:
                continue
            points = [project(x, y) for x, y in ring]
            origin_x, origin_y = points[0]
            area = math.fsum(
                (a[0] - origin_x) * (b[1] - origin_y)
                - (b[0] - origin_x) * (a[1] - origin_y)
                for a, b in zip(points, points[1:] + points[:1])
            )
            if area == 0:
                continue
            if (area > 0) != (index == 0):
                points.reverse()
            path.moveTo(*points[0])
            for point in points[1:]:
                path.lineTo(*point)
            path.closeSubpath()
    return path
