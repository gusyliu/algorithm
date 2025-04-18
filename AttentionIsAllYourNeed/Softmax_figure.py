import numpy as np
import matplotlib.pyplot as plt

# 定义Softmax函数
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # 防止数值溢出
    return exp_z / np.sum(exp_z)

# 计算Softmax函数对z_i的偏导数
def softmax_derivative(z, i, j):
    sigma = softmax(z)
    if i == j:
        return sigma[i] * (1 - sigma[i])
    else:
        return -sigma[i] * sigma[j]

# 设置输入范围
z1_values = np.linspace(-5, 5, 500)  # z1的变化范围
z2_fixed = 0  # 固定z2的值

# 计算Softmax输出和偏导数
softmax_outputs = []
derivatives = []

for z1 in z1_values:
    z = np.array([z1, z2_fixed])  # 输入向量 [z1, z2]
    softmax_output = softmax(z)[0]  # 取Softmax的第一个输出
    derivative = softmax_derivative(z, i=0, j=0)  # 偏导数 d(σ(z1))/dz1
    softmax_outputs.append(softmax_output)
    derivatives.append(derivative)

# 绘制图像
plt.figure(figsize=(12, 6))

# Softmax函数的输出
plt.subplot(1, 2, 1)
plt.plot(z1_values, softmax_outputs, label="Softmax(z1)", color="blue")
plt.title("Softmax Function Output")
plt.xlabel("z1")
plt.ylabel("Softmax(z1)")
plt.grid(True)
plt.legend()

# Softmax函数的导数
plt.subplot(1, 2, 2)
plt.plot(z1_values, derivatives, label="d(Softmax(z1))/dz1", color="red")
plt.title("Derivative of Softmax Function")
plt.xlabel("z1")
plt.ylabel("d(Softmax(z1))/dz1")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()