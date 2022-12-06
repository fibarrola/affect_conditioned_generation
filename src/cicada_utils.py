import pydiffvg
import torch
import random
import numpy as np
from torchvision import transforms

def treebranch_initialization(
    drawing,
    num_traces,
    drawing_area={'x0': 0, 'x1': 1, 'y0': 0, 'y1': 1},
    partition={'K1': 0.25, 'K2': 0.5, 'K3': 0.25},
):

    '''
    K1: % of curves starting from existing endpoints
    K2: % of curves starting from curves in K1
    K3: % of andom curves
    '''
    x0 = drawing_area['x0']
    x1 = drawing_area['x1']
    y0 = drawing_area['y0']
    y1 = drawing_area['y1']
    midpoint = [0.5 * (x0 + x1), 0.5 * (y0 + y1)]

    # Get all endpoints within drawing region
    starting_points = []
    starting_colors = []

    for trace in drawing.traces:
        # Maybe this is a tensor and I can't enumerate
        for k, point in enumerate(trace.shape.points):
            if k % 3 == 0:
                if (x0 < point[0] / drawing.canvas_width < x1) and (
                    y0 < (1 - point[1] / drawing.canvas_height) < y1
                ):
                    # starting_points.append(tuple([x.item() for x in point]))
                    starting_points.append(
                        (
                            point[0] / drawing.canvas_width,
                            point[1] / drawing.canvas_height,
                        )
                    )
                    starting_colors.append(trace.shape_group.stroke_color)

    # If no endpoints in drawing zone, we make everything random
    K1 = round(partition['K1'] * num_traces) if starting_points else 0
    K2 = round(partition['K2'] * num_traces) if starting_points else 0

    # Initialize Curves
    shapes = []
    shape_groups = []
    first_endpoints = []
    first_colors = []

    # Add random curves
    for k in range(num_traces):
        num_segments = random.randint(1, 3)
        num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
        points = []
        if k < K1:
            i0 = random.choice(range(len(starting_points)))
            p0 = starting_points[i0]
            color = torch.tensor(
                [
                    max(0.0, min(1.0, c + 0.3 * (random.random() - 0.5)))
                    for c in starting_colors[i0]
                ]
            )
        elif k < K2:
            i0 = random.choice(range(len(first_endpoints)))
            p0 = first_endpoints[i0]
            color = torch.tensor(
                [
                    max(0.0, min(1.0, c + 0.3 * (random.random() - 0.5)))
                    for c in first_colors[i0]
                ]
            )
        else:
            p0 = (
                torch.tensor(random.random() * (x1 - x0) + x0),
                torch.tensor(random.random() * (y1 - y0) + 1 - y1),
            )
            color = torch.rand(4)
        points.append(p0)

        theta0 = np.arctan2([midpoint[1] - (1 - p0[1])], [midpoint[0] - p0[0]]).item()

        for j in range(num_segments):
            radius = 0.05
            theta = np.random.normal(loc=theta0, scale=1)
            # substract the sin because y axis is upside down
            p1 = (
                p0[0] + radius * np.cos(theta),
                p0[1] - radius * np.sin(theta),
            )
            theta = np.random.normal(loc=theta0, scale=1)
            p2 = (
                p1[0] + radius * np.cos(theta),
                p1[1] - radius * np.sin(theta),
            )
            theta = np.random.normal(loc=theta0, scale=1)
            p3 = (
                p2[0] + radius * np.cos(theta),
                p2[1] - radius * np.sin(theta),
            )
            points.append(p1)
            points.append(p2)
            points.append(p3)
            p0 = p3

        if k < K1:
            first_endpoints.append(points[-1])
            first_colors.append(color)

        points = torch.tensor(points)
        points[:, 0] *= drawing.canvas_width
        points[:, 1] *= drawing.canvas_height
        path = pydiffvg.Path(
            num_control_points=num_control_points,
            points=points,
            stroke_width=torch.tensor(float(random.randint(1, 10)) / 2),
            is_closed=False,
        )
        shapes.append(path)
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([len(shapes) - 1]),
            fill_color=None,
            stroke_color=color,
        )
        shape_groups.append(path_group)

    return shapes, shape_groups


def get_augment_trans(canvas_width, normalize_clip=False):

    if normalize_clip:
        augment_trans = transforms.Compose(
            [
                transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
                transforms.RandomResizedCrop(canvas_width, scale=(0.7, 0.9)),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        augment_trans = transforms.Compose(
            [
                transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
                transforms.RandomResizedCrop(canvas_width, scale=(0.7, 0.9)),
            ]
        )

    return augment_trans