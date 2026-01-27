import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy

class PCGrad:
    def __init__(self, optimizer, reduction='mean'):
        self._optim = optimizer
        self._reduction = reduction
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        return self._optim.zero_grad()

    def step(self):
        return self._optim.step()

    def pc_backward(self, objectives):
        """
        objectives: A list of loss tensors (one per task)
        """
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        merged_grad = torch.zeros_like(grads[0])
        if self._reduction == 'mean':
            merged_grad = torch.stack(pc_grad).mean(0)
        elif self._reduction == 'sum':
            merged_grad = torch.stack(pc_grad).sum(0)
        return merged_grad

    def _set_grad(self, grads):
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _retrieve_grad(self):
        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).view(-1))
                    has_grad.append(torch.zeros_like(p).view(-1))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone().view(-1))
                has_grad.append(torch.ones_like(p).view(-1))
        return grad, shape, has_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat(grads)
        return flatten_grad

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = shape.numel()
            unflatten_grad.append(grads[idx:idx + length].view(shape))
            idx += length
        return unflatten_grad