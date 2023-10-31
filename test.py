import torch

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available")

# Create a tensor on the GPU
x = torch.randn(100, 100).cuda()

# Print the device of the tensor
print(x.device)


import torch

# Create a tensor on the CPU
x = torch.randn(100, 100)

# Move the tensor to the GPU
x = x.to('cuda:0')

# Perform some operations on the tensor
y = x + x

# Move the tensor back to the CPU
y = y.to('cpu')
