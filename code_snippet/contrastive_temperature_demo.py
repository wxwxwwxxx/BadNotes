import torch    
from torch import nn
# temperature demo
a = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]])
ce = nn.CrossEntropyLoss(reduction="none")
print("Input logits, for contrastive learning, it's the cosine similarity of two representation.")
print("temp=1")
print(ce(a, torch.tensor([0, 1, 2, 3], dtype=torch.int64)))
print("temp=0.5")
print(ce(a/0.5, torch.tensor([0, 1, 2, 3], dtype=torch.int64)))
print("temp=0.05")
print(ce(a/0.05, torch.tensor([0, 1, 2, 3], dtype=torch.int64)))
print("temp=2")
print(ce(a/2.0, torch.tensor([0, 1, 2, 3], dtype=torch.int64)))
print("temp=20")
print(ce(a/20.0, torch.tensor([0, 1, 2, 3], dtype=torch.int64)))
# Output:
# temp=1
# tensor([1.5425, 1.4425, 1.3425, 1.2425])
# temp=0.5
# tensor([1.7112, 1.5112, 1.3112, 1.1112])
# temp=0.05
# tensor([6.1451, 4.1451, 2.1451, 0.1451])
# temp=2
# tensor([1.4629, 1.4129, 1.3629, 1.3129])
# temp=20
# tensor([1.3938, 1.3888, 1.3838, 1.3788])
# Lower temperatures exacerbate disparities, while higher temperatures mitigate disparities.

# Numerical stability demo
a = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.11, 0.2, 0.3, 0.4]])
ce = nn.CrossEntropyLoss(reduction="none")
print("Numerical stability demo")
print("temp=0.5")
print(ce(a/0.5, torch.tensor([0, 0], dtype=torch.int64)))
print("temp=0.05")
print(ce(a/0.05, torch.tensor([0, 0], dtype=torch.int64)))
print("temp=0.005")
print(ce(a/0.005, torch.tensor([0, 0], dtype=torch.int64)))
print("temp=0.0005")
print(ce(a/0.0005, torch.tensor([0, 0], dtype=torch.int64)))
# Output:
# temp=1
# tensor([1.5425, 1.4425, 1.3425, 1.2425])
# temp=0.5
# tensor([1.7112, 1.5112, 1.3112, 1.1112])
# temp=0.05
# tensor([6.1451, 4.1451, 2.1451, 0.1451])
# temp=2
# tensor([1.4629, 1.4129, 1.3629, 1.3129])
# temp=20
# tensor([1.3938, 1.3888, 1.3838, 1.3788])
# When temperature gets higher, small disparities will cause large change in the output loss
# It will also cause numerical instability in the exp function inside the ce loss.
# Luckily, torch.nn.logsoftmax will help mitigate this:


m = nn.LogSoftmax(dim=1)
input = torch.randn(2, 3)
print(input)
# tensor([[-0.4541, 1.6703, -1.9000],
#         [0.9995, 0.7277, 1.3135]])
output = m(input)
print(output)
# tensor([[-2.2620, -0.1377, -3.7080],
#         [-1.1413, -1.4131, -0.8273]])
input = input * 100000
output = m(input)
print(output)
# tensor([[-212431.9062, 0.0000, -357025.3750],
#         [-31396.9297, -58581.7344, 0.0000]])
