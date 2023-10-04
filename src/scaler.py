import torch


class Scaler:
    def __init__(self, X, method='uniform'):
        """
        X: tensor in NxD, N:realisations, D:dimension
        method = 'uniform' | 'whiten' | 'normalize | none'
        """
        assert method in ['uniform', 'whiten', 'normalize', 'none']
        self.method = method
        if method == 'uniform':
            self.ub, _ = torch.max(X, dim=0)
            self.lb, _ = torch.min(X, dim=0)
        elif method == 'whiten':
            self.mu = torch.mean(X, dim=0)
            X = X - self.mu
            cov = torch.cov(torch.transpose(X, 1, 0))
            eigVals, eigVecs = torch.linalg.eigh(cov)
            self.W = eigVecs / (torch.sqrt(eigVals) + 1e-10)
            self.W_inv = torch.linalg.inv(self.W)
        elif method == 'normalize':
            self.mu = torch.mean(X, dim=0)
            self.sd = torch.std(X, dim=0)

    def scale(self, X):
        if self.method == 'uniform':
            # if torch.sum(torch.tensor(self.ub - self.lb)) == 0:
            #     return X
            return (X - self.lb) / (self.ub - self.lb + 1e-8)
        elif self.method == 'whiten':
            return torch.matmul((X - self.mu), self.W)
        elif self.method == 'normalize':
            return (X - self.mu) / self.sd
        elif self.method == 'none':
            return X

    def unscale(self, X):
        if self.method == 'uniform':
            return X * (self.ub - self.lb) + self.lb
        elif self.method == 'whiten':
            return torch.matmul(X, self.W_inv) + self.mu
        elif self.method == 'normalize':
            return X * self.sd + self.mu
        elif self.method == 'none':
            return X
