import torch.nn as nn


class PoseMetric(nn.Module):
    def __init__(self, num_cubes=3):
        super(PoseMetric, self).__init__()
        self.metric = nn.L1Loss()
        self.num_cubes = num_cubes

    def forward(self, pred, target):
        error = self.metric(pred, target)
        log_error = {
            "pose_error": error.item() * 100,
        }
        for i in range(self.num_cubes):
            cube_error = self.metric(
                pred[:, 0 + i * 3 : 3 + i * 3], target[:, 0 + i * 3 : 3 + i * 3]
            )
            x_error = self.metric(pred[:, 0 + i * 3], target[:, 0 + i * 3])
            y_error = self.metric(pred[:, 1 + i * 3], target[:, 1 + i * 3])
            z_error = self.metric(pred[:, 2 + i * 3], target[:, 2 + i * 3])
            log_error[f"error_cube{i}"] = x_error.item() * 100
            log_error[f"x_error_cube{i}"] = x_error.item() * 100
            log_error[f"y_error_cube{i}"] = y_error.item() * 100
            log_error[f"z_error_cube{i}"] = z_error.item() * 100
        return log_error


class VelMetric(nn.Module):
    def __init__(self, vel_key):
        super(VelMetric, self).__init__()
        self.metric = nn.L1Loss()
        self.vel_key = vel_key

    def forward(self, pred, target):
        error = self.metric(pred[self.vel_key], target[self.vel_key])
        x_error = self.metric(pred[self.vel_key][:, 0], target[self.vel_key][:, 0])
        y_error = self.metric(pred[self.vel_key][:, 1], target[self.vel_key][:, 1])
        z_error = self.metric(pred[self.vel_key][:, 2], target[self.vel_key][:, 2])
        log_error = {
            "vel_error": error.item() * 100,
            "x_error": x_error.item() * 100,
            "y_error": y_error.item() * 100,
            "z_error": z_error.item() * 100,
        }
        return log_error
