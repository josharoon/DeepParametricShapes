from random import randint
from unittest import TestCase

from unittest import TestCase
import torch
from torch import randperm

import templates
import utils
from templates import eye_templates,eye_templates2
n_loops=torch.tensor([1])
n_samples_per_curve=19

class Test(TestCase):
    def test_compute_curve_loss(self):
        def compute_curve_loss(points1, points2):
            batchSize, npoints, _ = points1.shape
            assert points1.shape == points2.shape

            min_loss = torch.tensor(float('inf')).to(points1.device)
            for j in range(npoints):
                # Shift points2 by j
                points2_shifted = torch.roll(points2, shifts=j, dims=1)

                # Calculate L1 norm and sum for each sample in the batch
                l1_norm = torch.sum(torch.abs(points1 - points2_shifted), dim=-1)
                batch_loss = torch.sum(l1_norm, dim=-1)

                # Keep the minimum loss across all shifts
                min_loss = torch.min(min_loss, batch_loss)

            return(min_loss / npoints).mean()

        topology = [15, 4, 4]
        # Create two identical sets of points
        points1Temp = torch.tensor(eye_templates[0]).reshape(-1, 2).unsqueeze(0)

        points1=utils.sample_points_from_curves(points1Temp, n_loops, topology, n_samples_per_curve)
        points2 = points1
        loss1 = compute_curve_loss(points1, points2)
        print(f"loss1: {loss1}")
        self.assertAlmostEqual(loss1.item(), 0.0, delta=1e-6)

        # Now randomly change the order of points in points2

        points1 = utils.sample_points_from_curves(points1Temp, n_loops, topology, n_samples_per_curve)
        shuffled_idx = randperm(points1Temp.shape[1])
        points2 = points1Temp[:, shuffled_idx, :]
        points2 = utils.sample_points_from_curves(points2, n_loops, topology, n_samples_per_curve)
        loss2 = compute_curve_loss(points1, points2)
        self.assertGreater(loss2.item(), 0.0)
        print(f"loss2 (points randomly shuffled): {loss2}")

        idx = torch.randint(0, points1Temp.shape[1], (2,))
        points1 = utils.sample_points_from_curves(points1Temp, n_loops, topology, n_samples_per_curve)
        points2 = points1Temp.clone()
        points2[:, idx[0], :], points2[:, idx[1], :] = points2[:, idx[1], :].clone(), points2[:, idx[0], :].clone()
        points2 = utils.sample_points_from_curves(points2, n_loops, topology, n_samples_per_curve)
        loss3 = compute_curve_loss(points1, points2)
        self.assertGreater(loss3.item(), 0.0)
        print(f"loss3 (2 points) shuffled): {loss3}")


        # get 2 different shapes but don't sample the points
        points1 = torch.tensor(eye_templates2[0]).reshape(-1, 2).unsqueeze(0)
        points2 = torch.tensor(eye_templates2[1]).reshape(-1, 2).unsqueeze(0)
        loss4 = compute_curve_loss(points1, points2)
        self.assertGreater(loss4.item(), 0.0)
        print(f"loss4 (2 shapes Unsampled curves): {loss4}")

        topology = [8, 4, 4]
        # Now create two different sets of points
        points1 = torch.tensor(eye_templates2[0]).reshape(-1, 2).unsqueeze(0)
        points1 = utils.sample_points_from_curves(points1, n_loops, topology, n_samples_per_curve)
        points2 = torch.tensor(eye_templates2[1]).reshape(-1, 2).unsqueeze(0)
        points2 = utils.sample_points_from_curves(points2, n_loops, topology, n_samples_per_curve)
        loss5 = compute_curve_loss(points1, points2)
        self.assertGreater(loss5.item(), 0.0)
        print(f"loss5 (2 shapes sampled curves): {loss5}")


