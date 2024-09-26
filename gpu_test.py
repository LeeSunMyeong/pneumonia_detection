import torch


torch.cuda.is_available()
var = torch.version

print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
print(var)