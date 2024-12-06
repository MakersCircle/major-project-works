import torch
print(f'Cuda Available: {torch.cuda.is_available()}')
x = torch.rand(5, 3)
print(x)
