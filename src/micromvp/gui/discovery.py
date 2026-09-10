"""Accumulate only obstacle-boundary portions exposed by visibility scans."""
import math


def discovered_obstacle_edges(obstacles, regions, tolerance=1e-6):
    """Return observed segments and fully observed obstacle indices.

    Visibility polygons contain free space. Their outer/hole boundaries share
    the obstacle edge intervals that were actually visible from each scan.
    """
    boundaries = []
    for outer, holes in regions:
        for polygon in [outer, *holes]:
            if len(polygon) >= 3:
                boundaries.extend(zip(polygon, polygon[1:] + polygon[:1]))
    observed = []
    complete = []
    for obstacle_index, obstacle in enumerate(obstacles):
        all_edges_seen = len(obstacle) >= 3
        for a, b in zip(obstacle, obstacle[1:] + obstacle[:1]):
            dx, dy = b[0]-a[0], b[1]-a[1]
            length = math.hypot(dx,dy)
            if length <= tolerance:
                continue
            intervals = []
            for c,d in boundaries:
                if (abs(dx*(c[1]-a[1])-dy*(c[0]-a[0])) > tolerance*length
                        or abs(dx*(d[1]-a[1])-dy*(d[0]-a[0])) > tolerance*length):
                    continue
                u=((c[0]-a[0])*dx+(c[1]-a[1])*dy)/(length*length)
                v=((d[0]-a[0])*dx+(d[1]-a[1])*dy)/(length*length)
                lo,hi=max(0.0,min(u,v)),min(1.0,max(u,v))
                if hi-lo > tolerance/length:
                    intervals.append((lo,hi))
            merged = []
            for lo,hi in sorted(intervals):
                if merged and lo <= merged[-1][1]+tolerance/length:
                    merged[-1]=(merged[-1][0],max(hi,merged[-1][1]))
                else:
                    merged.append((lo,hi))
            all_edges_seen &= sum(hi-lo for lo,hi in merged) >= 1-tolerance/length
            observed.extend([
                ((a[0]+lo*dx,a[1]+lo*dy),(a[0]+hi*dx,a[1]+hi*dy))
                for lo,hi in merged
            ])
        if all_edges_seen:
            complete.append(obstacle_index)
    return observed,complete
