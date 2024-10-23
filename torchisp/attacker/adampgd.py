
# Reference: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks," ICLR, 2018.
# Modified version using AdamW optimizer.

import torch
import torch.nn as nn

class RawSolver(nn.Module):
    """
    A helper class for optimizing the adversarial input using Adam.

    Arguments:
    - x (torch.Tensor): The initial input tensor to be optimized.

    This module allows the input tensor to be treated as a parameter for optimization, which is particularly
    useful in adversarial image generation tasks where the input itself is updated.
    """
    def __init__(self, x):
        super(RawSolver, self).__init__()
        params = x.clone().requires_grad_(True)
        self.params = nn.Parameter(params)
    
    def auto_clip(self):
        """
        Clamps the parameter values to the range [1e-6, 1] to ensure valid pixel values.
        """
        self.params.data = self.params.data.clamp(1e-6, 1)

    def forward(self):
        return self.params

class AdamPGD():
    """
    Adam-based Projected Gradient Descent (PGD) Attack.

    This class implements a modified version of PGD using the AdamW optimizer instead of the standard gradient descent.
    The attack is designed for generating adversarial examples for RAW image input, with the goal of optimizing
    the adversarial perturbation through more sophisticated gradient updates.

    Arguments:
    - predict (callable): A function or model that converts RAW RGGB input to RGB.
    - loss_fn (callable): The loss function used to calculate the adversarial loss.
    - lr (float): Learning rate for the AdamW optimizer. Defaults to 1e-4.
    - nb_iter (int): Number of iterations for the attack. Defaults to 1000.
    - eps_iter (float): Maximum perturbation for each iteration step. Defaults to 4/255.

    Reference:
    This implementation is inspired by Madry et al.'s PGD attack, modified to utilize AdamW for more efficient gradient optimization.
    """
    def __init__(self, predict, loss_fn, lr=1e-4, nb_iter=1000, eps_iter=4/255):
        self.predict = predict
        self.loss_fn = loss_fn
        self.lr = lr
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        
    def perturb(self, x, gt):
        """
        Perform the Adam-based PGD attack on the input tensor.

        Arguments:
        - x (torch.Tensor): The input tensor to be perturbed.
        - gt (torch.Tensor): The ground truth tensor used to calculate the adversarial loss.

        Returns:
        - torch.Tensor: The perturbed adversarial example.
        """
        x_adv = RawSolver(x)
        opt = torch.optim.AdamW(x_adv.parameters(), lr=self.lr)
        epoch = self.nb_iter // 8
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt, [epoch*4, epoch*6, epoch*7], gamma=0.25
        )

        for _ in range(self.nb_iter):
            opt.zero_grad()
            x_adv_rgb = self.predict(x_adv())
            loss = self.loss_fn(x_adv_rgb, gt)
            
            loss.backward()
            x_adv.params.grad = torch.nan_to_num(x_adv.params.grad)
            # Clip the gradient to avoid exploding gradients and ensure stability
            torch.nn.utils.clip_grad_norm_(x_adv.parameters(), self.eps_iter, norm_type='inf')

            opt.step()
            lr_scheduler.step()
            x_adv.auto_clip()
        
        return x_adv().detach()
