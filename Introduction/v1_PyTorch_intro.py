import torch

x = torch.tensor([1, 2, 3, 4, 5])
print(x)

# Create a float tensor
x_float = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
print(x_float)

# Create a 2x3 tensor filled with zeros
x_zeros = torch.zeros(2, 3)
print(x_zeros)

# Create a 2x3 tensor filled with ones
x_ones = torch.ones(2, 3)
print(x_ones)

# Create two tensors
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Add tensors
c = a + b
print(c)

# Subtract tensors
d = a - b
print(d)

# Multiply tensors element-wise
e = a * b
print(e)

x = torch.tensor(2.0, requires_grad=True)
y = x**2

# Calculate the gradient of y with respect to x
y.backward()
print(x.grad)

# Should be dy/dx = 2 * x = 4