import numpy as np
import torch

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)

print(f"Array: \n {data} \n")
print(f"Numpy Array: \n {np_array} \n")
print(f"PyTorch Tensor: \n {x_np} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

if torch.backends.mps.is_available():
    tensor = tensor.to('mps')
    print(f"Device tensor is stored on: {tensor.device}")

tensor = torch.ones(4, 4)
tensor[:, 1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)


print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
print(f"tensor * tensor \n {tensor * tensor}")


print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

print(tensor, "\n")
tensor.add_(5)
print(tensor)

# 각 클래스에 대한 예측 확률
predictions = torch.tensor([0.1, 0.2, 0.7])

# 가장 높은 확률을 가진 클래스의 인덱스 찾기
predicted_class = torch.argmax(predictions)
print(predicted_class)

a = torch.randn(4, 4)
print(a)
output = torch.argmax(a, dim=1) # 각 행에서 가장 큰 값의 인덱스를 반환
print(output)
output = torch.argmax(a, dim=0) # 각 열에서 가장 큰 값의 인덱스를 반환
print(output)