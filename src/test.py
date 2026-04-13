print("Testing the setup...")
import torch
print(torch.__file__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))