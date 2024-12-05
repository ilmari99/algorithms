import numpy as np
import itertools
import torch

# Define the vectors to match
v1 = torch.tensor([2., 3., 4.])
#v2 = torch.tensor([4., 3., 2., 1.])

# Generate all permutations of [1, 2, 3, 4]
all_permutations = list(itertools.permutations([0,2, 3, 4],3))
all_permutations.remove(tuple(v1.numpy()))
#all_permutations.remove(tuple(v2.numpy()))

# Convert permutations to torch tensors
permutations = [torch.tensor(p, dtype=torch.float32) for p in all_permutations]

w = torch.randn(3, requires_grad=True)

optimizer = torch.optim.Adam([w], lr=0.1)

delta = 1

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    w_dot_v1 = torch.dot(w, v1)
    #w_dot_v2 = torch.dot(w, v2)
    
    loss = 0
    for p in permutations:
        w_dot_p = torch.dot(w, p)
        # Compute loss for v1
        diff1 = delta - torch.abs(w_dot_v1 - w_dot_p)
        loss1 = torch.clamp(diff1, min=0)
        # Compute loss for v2
        #diff2 = delta - torch.abs(w_dot_v2 - w_dot_p)
        #loss2 = torch.clamp(diff2, min=0)
        loss += loss1# + loss2
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss {loss.item()}")

# After training
print(f"\nOptimized weights: {w.detach().numpy()}")
print(f"Dot product with v1: {torch.dot(w, v1).item()}")
#print(f"Dot product with v2: {torch.dot(w, v2).item()}")

# Verify that dot products with other permutations are at least delta apart
w_dot_v1 = torch.dot(w, v1).item()
#w_dot_v2 = torch.dot(w, v2).item()
violations = 0
for p in permutations:
    w_dot_p = torch.dot(w, p).item()
    if abs(w_dot_v1 - w_dot_p) < delta:
        violations += 1
        print(f"Violation for v1 with permutation {p.numpy()}, dot product difference: {abs(w_dot_v1 - w_dot_p)}")
if violations == 0:
    print("\nFound weights such that the dot products with v1 and v2 are unique among all permutations.")
else:
    print(f"\nNumber of violations: {violations}")
