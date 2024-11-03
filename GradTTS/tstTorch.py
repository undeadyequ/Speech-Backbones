import torch
import torch.nn as nn

print("ge")
x = torch.randn(3, 4)
mask = x.ge(0.5)

print("masked_select")
masked_x = torch.masked_select(x, mask)
masked_x1 = x.masked_fill(mask, 0)
print(x)
print(mask)
print(masked_x)
print(masked_x1)

print("argmax")
y = torch.tensor([[0, 0, 0], [0, 1, 1]])
print(torch.argmax(y))

print("nonzero")
a = torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
                            [0.0, 0.4, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0,-0.4]]
                           ), as_tuple=True)
print(a)




x = torch.randn(1, 1, 3, 4)
mask = x.ge(0.5)
a = torch.all(mask, dim=1)
print(mask)
print(a)


print("number of value:")
n_v = torch.count_nonzero(mask, )
print(mask)
print(n_v)



loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step()

print("")