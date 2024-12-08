import os
from random import shuffle
import sys
import torch
from tqdm import tqdm
import numpy as np

from dataclasses import dataclass

from torch.utils.tensorboard import SummaryWriter

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../src/")
)  # add the path to the DiffusionNet src

from pathlib import Path

from typing import List, Tuple

import diffusion_net


def _load_points(path: Path) -> Tuple[torch.Tensor, int]:
    with open(path) as f:
        count = int(f.readline())

        offset = 0

        result = torch.zeros((count, 3))

        for i in range(count):
            row = [float(e) for e in f.readline().split()]
            if i == 0:
                offset = int(row[0])
            result[i] = torch.tensor(row[1:])

        return result, offset


def _load_faces(path: Path) -> torch.Tensor:
    with open(path) as f:
        count = int(f.readline())

        result = torch.zeros((count, 3), dtype=torch.long)

        for i in range(count):
            result[i] = torch.tensor(
                [int(e) for e in f.readline().split()][1:4], dtype=torch.long
            )

        return result


def _load_pressure(path: Path) -> torch.Tensor:
    with open(path) as f:
        f.readline()
        f.readline()
        count = int(f.readline().split()[1])

        result = torch.zeros((count, 2))

        for i in range(count):
            result[i] = torch.tensor([float(e) for e in f.readline().split()][1:])

        return result


def _load_frequencies(path: Path) -> torch.Tensor:
    with open(path) as f:
        lines = f.readlines()
        result = torch.zeros(len(lines))

        for i, line in enumerate(lines):
            result[i] = float(line.split()[1])

        return result


@dataclass
class Shape:
    points: torch.Tensor
    faces: torch.Tensor
    frames: torch.Tensor
    mass: torch.Tensor
    L: torch.Tensor
    evals: torch.Tensor
    evecs: torch.Tensor
    gradX: torch.Tensor
    gradY: torch.Tensor
    device: str


@dataclass
class Evaluations:
    pressure: torch.Tensor
    frequency: float
    device: str


def load_data(path: Path, device: str = "cuda") -> Tuple[Shape, List[Evaluations]]:
    elements_folder = path / "EvaluationGrids" / "Default"
    outputs_folder = path / "NumCalc" / "source_1" / "be.out"

    pts, offset = _load_points(elements_folder / "Nodes.txt")
    faces = _load_faces(elements_folder / "Elements.txt") - offset

    pts = diffusion_net.geometry.normalize_positions(pts)

    frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(
        pts, faces, op_cache_dir="/tmp/diffusionnet/"
    )

    pts = pts.to(device)
    faces = faces.to(device)
    frames = frames.to(device)
    mass = mass.to(device)
    L = L.to(device)
    evals = evals.to(device)
    evecs = evecs.to(device)
    gradX = gradX.to(device)
    gradY = gradY.to(device)

    shape = Shape(pts, faces, frames, mass, L, evals, evecs, gradX, gradY, device)

    freqs = _load_frequencies(outputs_folder / "../Memory.txt")

    results = []
    for e in tqdm(os.listdir(outputs_folder), desc="Loading Data"):
        if os.path.isdir(outputs_folder / e) and e != Path("."):
            pressures = _load_pressure(outputs_folder / e / "pEvalGrid")
            pressures = pressures.to(device)
            results += [Evaluations(pressures, 100, device)]

    for r, f in zip(results, freqs):
        r.frequency = f

    return shape, results


def main(path: Path):
    c_in = 4
    c_out = 2  # complex real, imag
    width = 128
    device = "cuda"
    loss_type = "mse"

    model = diffusion_net.layers.DiffusionNet(
        C_in=c_in,
        C_out=c_out,
        C_width=width,
        outputs_at="vertices",
        last_activation=lambda x: torch.sigmoid(x),
    )

    shape, pressures = load_data(path, device)
    pressures = pressures[0:10]

    model.to("cuda")
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=250, gamma=0.5)

    epochs = 1000

    writer = SummaryWriter()

    max_freq = pressures[-1].frequency

    test_interval = 2500

    train_split = 0.7
    test_split = 0.2

    train_samples = max(int(len(pressures) * train_split), 10)
    test_samples = int(len(pressures) * test_split)
    val_samples = len(pressures) - train_samples - test_samples
    shuffle(pressures)

    p_min = 1
    p_max = -1

    for p in pressures[:train_samples]:
        p_max = max(p_max, torch.max(p.pressure).cpu().item())
        p_min = min(p_min, torch.min(p.pressure).cpu().item())

    print("max:", p_max, ",", "min:", p_min)

    for e in tqdm(range(epochs), desc="Epoch"):
        train_loss = 0
        for p in tqdm(pressures[:train_samples], desc="Sample", leave=False):
            optim.zero_grad()

            verts = torch.cat(
                (
                    shape.points,
                    torch.ones_like(shape.points[:, 0:1]) * p.frequency / max_freq,
                ),
                dim=1,
            )

            preds = model(
                verts,
                shape.mass,
                L=shape.L,
                evals=shape.evals,
                evecs=shape.evecs,
                gradX=shape.gradX,
                gradY=shape.gradY,
                faces=shape.faces,
            )

            if loss_type == "mse":
                loss = torch.nn.functional.mse_loss(
                    preds, (p.pressure - p_min) / (p_max - p_min)
                )
            elif loss_type == "smoothl1":
                loss = torch.nn.functional.smooth_l1_loss(
                    preds, (p.pressure - p_min) / (p_max - p_min)
                )
            elif loss_type == "mselinf":
                loss = torch.nn.functional.mse_loss(
                    preds, (p.pressure - p_min) / (p_max - p_min)
                ) + 0.5 * torch.max(
                    torch.abs(preds - (p.pressure - p_min) / (p_max - p_min))
                )
            else:
                raise RuntimeError

            loss.backward()

            optim.step()
            # scheduler.step()

            train_loss += loss.detach().clone().cpu().item()

        if e % test_interval == 0:
            with torch.no_grad():
                test_loss = 0
                for p in pressures[train_samples:-val_samples]:
                    verts = torch.cat(
                        (
                            shape.points,
                            torch.ones_like(shape.points[:, 0:1])
                            * p.frequency
                            / max_freq,
                        ),
                        dim=1,
                    )

                    preds = model(
                        verts,
                        shape.mass,
                        L=shape.L,
                        evals=shape.evals,
                        evecs=shape.evecs,
                        gradX=shape.gradX,
                        gradY=shape.gradY,
                        faces=shape.faces,
                    )

                    if loss_type == "mse":
                        test_loss += torch.nn.functional.mse_loss(
                            preds, (p.pressure - p_min) / (p_max - p_min)
                        )
                    elif loss_type == "smoothl1":
                        test_loss += torch.nn.functional.smooth_l1_loss(
                            preds, (p.pressure - p_min) / (p_max - p_min)
                        )
                    elif loss_type == "mselinf":
                        test_loss += torch.nn.functional.mse_loss(
                            preds, (p.pressure - p_min) / (p_max - p_min)
                        ) + 0.5 * torch.max(
                            torch.abs(preds - (p.pressure - p_min) / (p_max - p_min))
                        )
                    else:
                        raise RuntimeError

                    writer.add_scalar(
                        "Loss/test",
                        test_loss / len(pressures[train_samples:-val_samples]),
                        e,
                    )

        writer.add_scalar("Loss/train", train_loss / len(pressures[:train_samples]), e)


if __name__ == "__main__":
    main(Path("/home/sjaudia/Downloads/test_fem"))
