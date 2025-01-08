import copy
import random
from abc import abstractmethod
from typing import Dict, List, Tuple, Union

import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize, least_squares  

from min_norm_solvers import MinNormSolver, gradient_normalizers


class WeightMethod:
    def __init__(self, n_tasks: int, device: torch.device):
        super().__init__()
        self.n_tasks = n_tasks
        self.device = device

    @abstractmethod
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ],
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        representation: Union[torch.nn.parameter.Parameter, torch.Tensor],
        **kwargs,
    ):
        pass

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        last_shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        representation : shared representation
        kwargs :

        Returns
        -------
        Loss, extra outputs
        """
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )
        loss.backward()
        return loss, extra_outputs

    def __call__(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        return self.backward(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            **kwargs,
        )

    def parameters(self) -> List[torch.Tensor]:
        """return learnable parameters"""
        return []


class NashMTL(WeightMethod):
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        max_norm: float = 1.0,
        update_weights_every: int = 1,
        optim_niter=20,
    ):
        super().__init__(
            n_tasks=n_tasks,
            device=device,
        )

        self.optim_niter = optim_niter
        self.update_weights_every = update_weights_every
        self.max_norm = max_norm

        self.prvs_alpha_param = None
        self.normalization_factor = np.ones((1,))
        self.init_gtg = self.init_gtg = np.eye(self.n_tasks)
        self.step = 0.0
        self.prvs_alpha = np.ones(self.n_tasks, dtype=np.float32)

    def _stop_criteria(self, gtg, alpha_t):
        return (
            (self.alpha_param.value is None)
            or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
            or (
                np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value)
                < 1e-6
            )
        )

    def solve_optimization(self, gtg: np.array):
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t

            try:
                self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except:
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(gtg, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha

    def _calc_phi_alpha_linearization(self):
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
        return phi_alpha

    def _init_optim_problem(self):
        self.alpha_param = cp.Variable(shape=(self.n_tasks,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(
            shape=(self.n_tasks,), value=self.prvs_alpha
        )
        self.G_param = cp.Parameter(
            shape=(self.n_tasks, self.n_tasks), value=self.init_gtg
        )
        self.normalization_factor_param = cp.Parameter(
            shape=(1,), value=np.array([1.0])
        )

        self.phi_alpha = self._calc_phi_alpha_linearization()

        G_alpha = self.G_param @ self.alpha_param
        constraint = []
        for i in range(self.n_tasks):
            constraint.append(
                -cp.log(self.alpha_param[i] * self.normalization_factor_param)
                - cp.log(G_alpha[i])
                <= 0
            )
        obj = cp.Minimize(
            cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param
        )
        self.prob = cp.Problem(obj, constraint)

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        """

        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :

        Returns
        -------

        """

        extra_outputs = dict()
        if self.step == 0:
            self._init_optim_problem()

        if (self.step % self.update_weights_every) == 0:
            self.step += 1

            grads = {}
            for i, loss in enumerate(losses):
                grad = list(
                    torch.autograd.grad(
                        loss,
                        shared_parameters,
                        retain_graph=True,
                        allow_unused=True
                    )
                )
                # grad = torch.cat([torch.flatten(grad) for grad in g])
                grad = [
                    g.flatten() if g is not None else torch.zeros_like(p, device=self.device).flatten()
                    for g, p in zip(grad, shared_parameters)
                ]   
                grads[i] = grad

            G = torch.stack(tuple(v for v in grads.values()))
            GTG = torch.mm(G, G.t())

            self.normalization_factor = (
                torch.norm(GTG).detach().cpu().numpy().reshape((1,))
            )
            GTG = GTG / self.normalization_factor.item()
            alpha = self.solve_optimization(GTG.cpu().detach().numpy())
            alpha = torch.from_numpy(alpha)

        else:
            self.step += 1
            alpha = self.prvs_alpha
        #alpha = alpha / (torch.sum(alpha**2)**(1/2)+1e-6)
        #print(alpha, losses)
        #alpha = [i if losses[idx] > -np.log(0.99) else min(i, 1) for idx, i in enumerate(alpha)]
        weighted_loss = sum([losses[i] * alpha[i] for i in range(len(alpha))])
        extra_outputs["weights"] = alpha
        return weighted_loss, extra_outputs

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        last_shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[Dict, None]]:
        #mags = [abs(i.item())+1e-8 for i in losses]
        #losses = [i/mags[idx] for idx, i in enumerate(losses)]
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            **kwargs,
        )
        # loss.backward()

        # make sure the solution for shared params has norm <= self.eps
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)

        return loss, extra_outputs

class FairGrad(WeightMethod):
    """
    FairGrad.

    This method is proposed in `Fair Resource Allocation in Multi-Task Learning (ICML 2024)
    <https://openreview.net/forum?id=KLmWRMg6nL>`_ and implemented by modifying from the 
    `official PyTorch implementation <https://github.com/OptMN-Lab/fairgrad>`_.
    """
    def __init__(self, n_tasks: int, device: torch.device, **kwargs):
        super().__init__(n_tasks, device, **kwargs)
        self.kwargs = kwargs
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ],
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        representation: Union[torch.nn.parameter.Parameter, torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the weighted loss using FairGrad.

        Parameters
        ----------
        losses : torch.Tensor
            Tensor of task-specific losses.
        shared_parameters : Union[List[torch.nn.parameter.Parameter], torch.Tensor]
        task_specific_parameters : Union[List[torch.nn.parameter.Parameter], torch.Tensor]
        last_shared_parameters : Union[List[torch.nn.parameter.Parameter], torch.Tensor]
        representation : Union[torch.nn.parameter.Parameter, torch.Tensor]
        kwargs : dict
            Additional arguments, including FairGrad_alpha.

        Returns
        -------
        loss : torch.Tensor
            Weighted loss for backward computation.
        extra_outputs : dict
            Additional outputs, including task weights.
        """
        alpha = self.kwargs.get("FairGrad_alpha", 0.8)

        if representation is not None:
            raise ValueError("FairGrad does not support representation gradients (rep_grad=True)")

        # Compute gradient matrix G
        grads = self._compute_grad(losses, shared_parameters)
        GTG = torch.mm(grads, grads.t())

        # Solve the FairGrad optimization problem using Torch
        x = torch.ones(self.n_tasks, device=self.device, requires_grad=True) / self.n_tasks
        A = GTG.data.cpu().numpy()

        def objfn(x):
            return np.dot(A, x) - np.power(1 / x, 1 / alpha)

        res = least_squares(objfn, x.cpu().detach().numpy(), bounds=(0, np.inf))
        w_cpu = res.x


        weights = torch.Tensor(w_cpu).to(self.device)

        # Compute weighted loss
        try:
            weighted_loss = sum([losses[i] * weights[i] for i in range(len(weights))])
        except:
            print("FairGrad failed")
            print(weights.shape)
            print(losses)
        # weighted_loss = torch.sum(weights * losses)

        # Return loss and extra outputs
        extra_outputs = {"weights": weights.cpu().numpy()}
        return weighted_loss, extra_outputs

    def _compute_grad(self, losses: torch.Tensor, parameters: List[torch.nn.parameter.Parameter]):
        """
        Compute gradients of the losses w.r.t. the shared parameters.

        Parameters
        ----------
        losses : torch.Tensor
        parameters : List[torch.nn.parameter.Parameter]

        Returns
        -------
        grads : torch.Tensor
            Matrix of gradients (n_tasks x n_parameters).
        """
        grads = []
        for i in range(self.n_tasks):
            grad = torch.autograd.grad(
                losses[i], parameters, retain_graph=True, create_graph=False, allow_unused=True
            )
            grad = [
                g.flatten() if g is not None else torch.zeros_like(p, device=self.device).flatten()
                for g, p in zip(grad, parameters)
            ]
            grad_flat = torch.cat(grad)
            grads.append(grad_flat)

        return torch.stack(grads)

