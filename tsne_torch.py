import torch

def pairwise_distance(data):
    n = data.size(0)
    dist = torch.pow(data, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, data, data.t())
    dist = dist.clamp(min=1e-10).sqrt()  
    return dist

def cal_P(dist, sigmas=None):
    if sigmas is None:
        sigmas = torch.ones(dist.size(0)).cuda()
    else:
        sigmas = sigmas.view(-1, 1)
    exp_dist = torch.exp(-dist / (2 * sigmas * sigmas))
    P = exp_dist / exp_dist.sum(dim=1).view(-1, 1)
    return P

def cal_Q(Y):
    dist = pairwise_distance(Y)
    Q = 1.0 / (1.0 + dist)
    Q = Q / Q.sum(dim=1).view(-1, 1)
    return Q

def t_sne(X, num_iterations=1000, learning_rate=0.1):
    # Initialize Y with random values
    Y = torch.rand(X.size()).cuda()
    Y.requires_grad_()
    optimizer = torch.optim.SGD([Y], lr=learning_rate)

    P = cal_P(pairwise_distance(X))

    for i in range(num_iterations):
        optimizer.zero_grad()
        Q = cal_Q(Y)
        loss = P * torch.log(P / Q)
        loss = loss.sum()
        loss.backward(retain_graph=True)
        optimizer.step()
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")
    return Y.clone().detach()
