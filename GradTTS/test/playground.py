import torch
import torch
import torch.nn as nn
def test1():

    a = torch.tensor([1.], requires_grad=True)
    y = torch.zeros((10))
    gt = torch.zeros((10))

    y[0] = a
    y[1] = y[0] * 2
    #y.retain_grad()
    loss = torch.sum((y-gt) ** 2)
    loss.backward()
    print(y.grad)


def test2():


    criterion = nn.MSELoss()

    a = torch.tensor([1., 2.], requires_grad=True)
    b = torch.tensor([1., 2.], requires_grad=True)
    a_copy = torch.clone(a)
    y_gd = torch.tensor([1., 1.])

    optim = torch.optim.SGD([a, b], lr=1e-2, momentum=0.9)

    y = a ** 2 + b * 2
    loss = criterion(y, y_gd)

    loss.backward()

    print("a, b grad:", a.grad, b.grad)
    a_beforeGrad = b.clone()
    b_beforeGrad = b.clone()
    print("a, b:", a_beforeGrad, b_beforeGrad)

    optim.step()
    print("If a is same:{}".format(torch.equal(a_copy, a)))

    print("a, b grad:", a.grad, b.grad)
    print("a, b:", a, b)
    assert a[0] == a_beforeGrad[0] - 0.01 * a.grad[0]
    assert b[0] == b_beforeGrad[0] - 1e-2 * b.grad[0]


if __name__ == '__main__':
    test2()