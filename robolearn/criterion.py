import torch.nn as nn


class BCCriterion(nn.Module):
    def __init__(self, lam):
        super(BCCriterion, self).__init__()
        self.vel_criterion = nn.MSELoss()
        self.grip_criterion = nn.BCELoss()
        self.lam = lam

    def forward(self, pred, target):
        loss_vel = self.vel_criterion(pred["vel"], target["vel"])
        log_loss = {"loss_vel": loss_vel.item()}
        loss = loss_vel
        if "grip" in target.keys():
            loss_grip = self.grip_criterion(pred["grip"], target["grip"])
            loss = self.lam * loss + (1 - self.lam) * loss_grip
            log_loss["loss_grip"] = loss_grip.item()
        return loss, log_loss


class PoseCriterion(nn.Module):
    def __init__(self):
        super(PoseCriterion, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred, target):
        loss = self.criterion(pred["pose"], target["pose"])
        log_loss = {"loss_pose": loss.item()}
        return loss, log_loss
