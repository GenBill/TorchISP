# Reference: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks," International Conference on Learning Representations (ICLR), 2018.

import torch
import torch.nn as nn
import numpy as np

def _model_freeze(model) -> None:
    for param in model.parameters():
        param.requires_grad = False

def _model_unfreeze(model) -> None:
    for param in model.parameters():
        param.requires_grad = True

class TargetLinfPGD():
    """
    Targeted Projected Gradient Descent (PGD) Attack.

    This class implements a targeted PGD attack, primarily designed for generating adversarial RAW images.
    It freezes the model to prevent parameter updates during attack and operates in evaluation mode.
    The attack works by iteratively perturbing the input to move it towards a specified target, making it useful
    for adversarial RAW image generation tasks.

    Arguments:
    - predict (nn.Module): The model to attack, which will be used to generate adversarial examples.
    - loss_fn (callable, optional): The loss function used to calculate the adversarial loss. Defaults to CrossEntropyLoss.
    - eps (float): Maximum perturbation allowed for the adversarial examples.
    - eps_iter (float): Step size for each iteration of the attack.
    - nb_iter (int): Number of attack iterations.

    Usage:
    This attack is suitable for scenarios where targeted adversarial examples are required, particularly for RAW image generation.
    It is recommended to use specific initialization (`rand_init=False`) and targeted attack mode (`target=True`) to achieve optimal results.
    """
    def __init__(self, predict, loss_fn=None, eps=32/256, eps_iter=1/256, nb_iter=100, rand_init=False, targeted=True):
        super(TargetLinfPGD, self).__init__()
        # Arguments of PGD
        self.device = next(predict.parameters()).device

        self.predict = predict
        self.loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss()
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        
        self.rand_init = rand_init
        self.targeted = targeted

        _model_freeze(self.predict)
        self.predict.eval()

    def perturb(self, data, target):

        x_adv = data.clone().detach().to(self.device)

        if self.rand_init:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for _ in range(self.nb_iter):
            x_adv.requires_grad_()
            output = self.predict(x_adv)
            
            with torch.enable_grad():
                loss_adv = self.loss_fn(output, target)
                if not self.targeted:
                    loss_adv = -loss_adv
            
            x_adv.grad = None  # Clear gradients before backward pass
            loss_adv.backward()
            eta = self.eps_iter * x_adv.grad.sign()
            x_adv = x_adv.detach() - eta
            x_adv = torch.min(torch.max(x_adv, data - self.eps), data + self.eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv
