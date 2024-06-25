import paddle

print('飞桨框架内置模型：', paddle.vision.models.__all__)

# 飞桨框架内置模型： ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext50_64x4d', 'resnext101_32x4d', 'resnext101_64x4d', 'resnext152_32x4d', 'resnext152_64x4d', 'wide_resnet50_2', 'wide_resnet101_2', 'VGG', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'MobileNetV1', 'mobilenet_v1', 'MobileNetV2', 'mobilenet_v2', 'MobileNetV3Small', 'MobileNetV3Large', 'mobilenet_v3_small', 'mobilenet_v3_large', 'LeNet', 'DenseNet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'densenet264', 'AlexNet', 'alexnet', 'InceptionV3', 'inception_v3', 'SqueezeNet', 'squeezenet1_0', 'squeezenet1_1', 'GoogLeNet', 'googlenet', 'ShuffleNetV2', 'shufflenet_v2_x0_25', 'shufflenet_v2_x0_33', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'shufflenet_v2_swish']

# 模型组网并初始化网络
lenet = paddle.vision.models.LeNet(num_classes=10)

# 可视化模型组网结构和参数
paddle.summary(lenet, (1, 1, 28, 28))

from paddle import nn

# 使用 paddle.nn.Sequential 构建 LeNet 模型
lenet_Sequential = nn.Sequential(
    nn.Conv2D(1, 6, 3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2D(2, 2),
    nn.Conv2D(6, 16, 5, stride=1, padding=0),
    nn.ReLU(),
    nn.MaxPool2D(2, 2),
    nn.Flatten(),
    nn.Linear(400, 120),
    nn.Linear(120, 84),
    nn.Linear(84, 10)
)
# 可视化模型组网结构和参数
paddle.summary(lenet_Sequential,(1, 1, 28, 28))


# 使用 Subclass 方式构建 LeNet 模型
class LeNet(nn.Layer):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        # 构建 features 子网，用于对输入图像进行特征提取
        self.features = nn.Sequential(
            nn.Conv2D(
                1, 6, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(2, 2),
            nn.Conv2D(
                6, 16, 5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2D(2, 2))
        # 构建 linear 子网，用于分类
        if num_classes > 0:
            self.linear = nn.Sequential(
                nn.Linear(400, 120),
                nn.Linear(120, 84),
                nn.Linear(84, num_classes)
            )
    # 执行前向计算
    def forward(self, inputs):
        x = self.features(inputs)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.linear(x)
        return x
lenet_SubClass = LeNet()

# 可视化模型组网结构和参数
params_info = paddle.summary(lenet_SubClass,(1, 1, 28, 28))
print(params_info)




x = paddle.uniform((2, 3, 8, 8), dtype='float32', min=-1., max=1.)

conv = nn.Conv2D(3, 6, (3, 3), stride=2)  # 卷积层输入通道数为3，输出通道数为6，卷积核尺寸为3*3，步长为2
y = conv(x) # 输入数据X
y = y.numpy()
print(y.shape)




x = paddle.uniform((2, 3, 8, 8), dtype='float32', min=-1., max=1.)

pool = nn.MaxPool2D(3, stride=2) # 池化核尺寸为3*3，步长为2
y = pool(x) #输入数据x
y = y.numpy()
print(y.shape)



x = paddle.uniform((2, 6), dtype='float32', min=-1., max=1.)
linear = paddle.nn.Linear(6, 4)
y = linear(x)
print(y.shape)



x = paddle.to_tensor([-2., 0., 1.])
relu = paddle.nn.ReLU()
y = relu(x)
print(y)


for name, param in lenet.named_parameters():
    print(f"Layer: {name} | Size: {param.shape}")