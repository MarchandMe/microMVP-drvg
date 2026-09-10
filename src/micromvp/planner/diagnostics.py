"""Inspect native RVG angular-layer geometry from a saved scene."""
import math
from types import SimpleNamespace

import numpy as np

from micromvp.core.models import WorkspaceConfig
from .dynamic_rvg import DynamicRVGSession, DynamicRVGSettings, pad_obstacles, _polygons_intersect


class SolverInspector:
    def __init__(self, scene):
        self.scene = scene
        self.workspace = WorkspaceConfig(**scene["workspace"])
        self.pose = tuple(scene["pose"])
        self.settings = DynamicRVGSettings(**scene["settings"])
        self.session = DynamicRVGSession(
            self.workspace,
            pad_obstacles(scene["obstacles"], scene["obstacle_padding_cm"]),
            self.settings,
        )
        self.session.initialize(self.pose, self.pose)
        saved_scan = scene.get("native_observation")
        if saved_scan is None:
            self.observation = self.session.scan()
        else:
            def polygon(points):
                return self.session._rvg.polygon(
                    [self.session._rvg.vertex(x,y) for x,y in points], False
                )
            self.observation = SimpleNamespace(
                success=saved_scan["success"],
                outerBoundary=polygon(saved_scan["outer"]),
                holes=[polygon(points) for points in saved_scan["holes"]],
            )
        if not self.observation.success:
            raise RuntimeError("Native visibility scan failed at the captured pose")
        self._layers = {}

    def points(self, polygon):
        return self.session._polygon_points(polygon)

    def layer(self, index):
        index %= self.settings.resolution
        if index not in self._layers:
            step = 2 * math.pi / self.settings.resolution
            layer = self.session._rvg.Layer(index * step, (index+1) * step, False, False)
            # GraphMerge uses precisely this scan border and these scan holes.
            layer.buildVisibilityGraph(
                self.session._native_robot,
                self.observation.outerBoundary,
                list(self.observation.holes),
            )
            self._layers[index] = layer
        return self._layers[index]

    def snapshot(self, heading):
        heading %= 360
        index = min(self.settings.resolution-1, int(heading * self.settings.resolution / 360))
        layer = self.layer(index)
        pose = (self.pose[0], self.pose[1], heading)
        bbox = self.points(layer.getRobotBBox())
        placed_bbox = [(px+pose[0],py+pose[1]) for px,py in bbox]
        outside = any(px < 0 or px > self.workspace.width or py < 0 or py > self.workspace.height
                      for px,py in placed_bbox)
        intersected = [
            i for i,poly in enumerate(self.session._native_obstacles)
            if _polygons_intersect(placed_bbox,self.points(poly))
        ]
        legal = bool(layer.legalConfig(self.session._vertex(pose)))
        cause = (
            "angular bounding polygon crosses workspace boundary" if outside else
            f"angular bounding polygon intersects solver obstacle(s) {intersected}" if intersected else
            "native layer rejects axle within visible-space model" if not legal else
            "valid"
        )
        return {
            "pose": pose, "layer_index": index,
            "cause": cause,
            "bbox_outside_workspace": outside,
            "bbox_intersected_obstacles": intersected,
            "unscaled_pose_clear": self.session.pose_is_valid(pose,robot_geometry_scale=1.0),
            "scaled_pose_clear": self.session.pose_is_valid(pose,robot_geometry_scale=self.settings.robot_geometry_scale),
            "theta_lower_deg": math.degrees(layer.getThetaLb()),
            "theta_upper_deg": math.degrees(layer.getThetaUb()),
            "native_legal": bool(layer.legalConfig(self.session._vertex(pose))),
            "layer_feasible": bool(layer.isFeasible()),
            "robot_bbox": self.points(layer.getRobotBBox()),
            "grown_obstacles": [self.points(p) for p in layer.getGrownObs()],
            "free_pockets": [self.points(p) for p in layer.getHoles()],
            "shrunken_border": self.points(layer.getShrinkedBorder()),
        }

    def legal_grid(self, heading, spacing=.6):
        layer = self.layer(int((heading % 360) * self.settings.resolution / 360))
        xs = np.linspace(0, self.workspace.width, max(2, math.ceil(self.workspace.width/spacing)))
        ys = np.linspace(0, self.workspace.height, max(2, math.ceil(self.workspace.height/spacing)))
        legal = np.array([
            [layer.legalConfig(self.session._vertex((float(x),float(y),heading))) for x in xs]
            for y in ys
        ], dtype=bool)
        return xs, ys, legal

    def plot(self, heading=None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon, Patch
        from matplotlib.colors import ListedColormap

        heading = self.pose[2] if heading is None else heading
        data = self.snapshot(heading)
        fig, axes = plt.subplots(2,2,figsize=(16,10),layout="constrained")
        x,y,_ = data["pose"]
        color = "#15945e" if data["native_legal"] else "#d22f3e"
        physical = DynamicRVGSession.physical_robot_geometry(self.workspace)
        planning = [(px*self.settings.robot_geometry_scale,py*self.settings.robot_geometry_scale)
                    for px,py in self.session._base_robot_geometry]
        angle = math.radians(heading)
        c,s = math.cos(angle),math.sin(angle)
        def rotated(points):
            return [(px*c-py*s,px*s+py*c) for px,py in points]
        def shifted(points):
            return [(px+x,py+y) for px,py in points]
        def polygon(ax, points, edge, fill="none", alpha=1, width=1.5, label=None):
            if len(points) >= 3:
                ax.add_patch(Polygon(points,closed=True,edgecolor=edge,facecolor=fill,
                                     alpha=alpha,linewidth=width,label=label))
        border = self.points(self.session._native_border)
        seen = self.points(self.observation.outerBoundary)
        obstacles = [self.points(p) for p in self.session._native_obstacles]
        def world(ax, zoom=False):
            polygon(ax,border,"#25485c",width=2,label="Workspace")
            polygon(ax,seen,"#c69220","#f6d875",.25,label="Native visible area")
            for i,obstacle in enumerate(obstacles):
                polygon(ax,obstacle,"#3f4a54","#9aa6af",.8,label="Solver obstacle" if i==0 else None)
            polygon(ax,shifted(rotated(physical)),"#171717",width=2,label="Physical body")
            polygon(ax,shifted(rotated(planning)),"#087fa6",width=2,label="Scaled planning footprint")
            # Native bounding polygon is already rotated over the angle interval.
            polygon(ax,shifted(data["robot_bbox"]),"#cc315a",width=2,label="Native layer bound")
            ax.plot(x,y,"x",color=color,markersize=10,markeredgewidth=3,label="Wheel axle")
            ax.arrow(x,y,4*c,4*s,width=.12,color=color,length_includes_head=True)
            if zoom:
                r=max(math.hypot(px,py) for px,py in data["robot_bbox"])+5
                ax.set_xlim(x-r,x+r);ax.set_ylim(y-r,y+r)
            else:
                ax.set_xlim(-2,self.workspace.width+2);ax.set_ylim(-2,self.workspace.height+2)
        world(axes[0,0])
        axes[0,0].set_title("Environment used by the solver")
        axes[0,0].legend(loc="best",fontsize=8)

        ax=axes[0,1]
        polygon(ax,rotated(physical),"#171717",width=2,label="Physical body")
        polygon(ax,rotated(planning),"#087fa6",width=2,label=f"Planning footprint ×{self.settings.robot_geometry_scale:g}")
        polygon(ax,data["robot_bbox"],"#cc315a","#ed9aad",.35,width=2,label="Native angular-layer bound")
        ax.plot(0,0,"+",color="#171717",markersize=16,markeredgewidth=2,label="RVG rotation centre / axle")
        ax.axhline(0,color="#ccd4da",linewidth=.7);ax.axvline(0,color="#ccd4da",linewidth=.7)
        r=max(math.hypot(px,py) for px,py in data["robot_bbox"])+1
        ax.set_xlim(-r,r);ax.set_ylim(-r,r)
        ax.set_title(f"Footprint about axle: {data['theta_lower_deg']:.0f}–{data['theta_upper_deg']:.0f}° layer")
        ax.legend(loc="best",fontsize=8)

        ax=axes[1,0]
        xs,ys,legal=self.legal_grid(heading)
        ax.imshow(legal.astype(int),origin="lower",extent=(xs[0],xs[-1],ys[0],ys[-1]),
                  cmap=ListedColormap(["#f4d4d6","#cdebdc"]),vmin=0,vmax=1,interpolation="nearest")
        polygon(ax,data["shrunken_border"],"#087fa6",width=2,label="Native shrunken border")
        for i,poly in enumerate(data["grown_obstacles"]):
            polygon(ax,poly,"#b73548",width=1.5,label="Native grown obstacles" if i==0 else None)
        for i,poly in enumerate(data["free_pockets"]):
            polygon(ax,poly,"#337c5a",width=1,label="Native free pockets" if i==0 else None)
        ax.plot(x,y,"x",color=color,markersize=12,markeredgewidth=3)
        ax.set_title("Configuration space: native legalConfig() for the axle")
        handles,labels=ax.get_legend_handles_labels()
        handles=[Patch(facecolor="#cdebdc",label="Allowed axle position"),
                 Patch(facecolor="#f4d4d6",label="Forbidden axle position")]+handles
        ax.legend(handles=handles,loc="best",fontsize=8)
        world(axes[1,1],zoom=True)
        axes[1,1].set_title("Current-pose close-up")
        for ax in axes.flat:
            ax.set_aspect("equal",adjustable="box")
            ax.set_xlabel("x (cm)");ax.set_ylabel("y (cm)")
            ax.grid(alpha=.15)
        verdict="VALID" if data["native_legal"] else "INVALID"
        fig.suptitle(
            f"DRVG snapshot • axle ({x:.2f}, {y:.2f}) cm • heading {heading:.1f}° • "
            f"native layer: {verdict}\n"
            f"Footprint: {self.settings.robot_footprint} ×{self.settings.robot_geometry_scale:g} | "
            f"Obstacle padding: {self.scene['obstacle_padding_cm']:g} cm | "
            f"Layer {data['layer_index']+1}/{self.settings.resolution}",
            fontsize=15,color=color,
        )
        return fig,data
