from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as tvm
from torchvision.models import resnet
from utils.functions import (check_basic_block_structure,
                             check_bottleneck_structure)


class SequentialModel(nn.Module):
    def __init__(self):
        super(SequentialModel, self).__init__()
        self.layers = []
        self.dims = [[3, 32, 32], 10]
        # IMPORTANT: include this line at the end of init of each daughter class
        # self.layer_input_shapes = self.get_layer_input_shapes()

    def forward(self, x):
        return self.layers(x)

    def forward_up_to_k_layer(self, x, k):
        out = x
        for i, layer in enumerate(self.layers):
            if i >= k:
                return out
            out = layer(out)
        return out

    def get_layer_input_shapes(self):
        """This function calculates the input shapes for each layer.
        This is required to compute the proper upper bound for CNN and ResNet 
        layers.

        Returns
        -------
            The list of input shapes for each layer in `layers` and the shape of
            the output layer. Use `input_shapes[i]` to get the input shape for 
            layer with index `i`. If the layer itself is a Sequential layer or a
            ResNet BasicBlock, it will have nested input shapes.
        """
        return SequentialModel._get_layer_input_shapes(
            self.layers, self.dims[0])[0]

    def _get_layer_input_shapes(layers: Sequence,
                                inp_shape: list[int] | int) -> list:
        # check for 1D or nD input
        if isinstance(inp_shape, int):
            input_shape = [inp_shape]
        else:
            input_shape = inp_shape

        # here batchsize = 1
        t = torch.zeros([1] + input_shape)
        input_shapes = []
        last_shape = input_shape

        for layer in layers:
            if isinstance(layer, nn.Sequential):
                res, last_shape = SequentialModel._get_layer_input_shapes(
                    layer, last_shape)
                input_shapes.append(res)
                t = layer(t)
            elif isinstance(layer,
                            resnet.BasicBlock) or isinstance(layer,
                                                             resnet.Bottleneck):
                # treat the resnet block case separately
                if isinstance(layer, resnet.BasicBlock):
                    assert check_basic_block_structure(layer)
                else:
                    assert check_bottleneck_structure(layer)

                # do it manually here
                # input of prev layer
                resnet_shapes = [last_shape]

                t = layer.conv1(t)
                last_shape = list(t.shape)[1:]

                # input to conv2
                resnet_shapes.append(last_shape)

                # skip batch norms and relu layers as they do not change the
                # shape
                t = layer.conv2(t)
                last_shape = list(t.shape)[1:]

                # add 2nd convolution output for the bottleneck module case
                if isinstance(layer, resnet.Bottleneck):
                    # input to conv3
                    resnet_shapes.append(last_shape)

                    t = layer.conv3(t)
                    last_shape = list(t.shape)[1:]

                # the downsample layer will have the last_shape input shape
                # no need to propogate further as there is no change in shape
                # in other layers
                input_shapes.append(resnet_shapes)
            else:
                # any default layer
                input_shapes.append(last_shape)
                t = layer(t)
                last_shape = list(t.shape)[1:]

        return input_shapes, last_shape


class ReLU_Net(SequentialModel):
    def __init__(
        self,
        widths=[16, 64],
        dims=[40, 10],
        bias=False,
        leaky_output=False,
    ):
        # output from the example
        # layers=nn.Sequential(
        #     nn.Linear(40, 16, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(16, 64, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(64, 10, bias=False),
        #     nn.ReLU()
        # )
        super(ReLU_Net, self).__init__()

        self.dims = dims

        num_layers = 2 * (len(widths) + 1)
        all_dims = [dims[0]] + widths + [dims[1]]
        layers = []

        for i in range(num_layers):
            if i % 2 == 0:
                j = i // 2
                layers.append(
                    nn.Linear(all_dims[j], all_dims[j + 1], bias=bias))
            else:
                layers.append(nn.ReLU())

        if leaky_output:
            layers[-1] = nn.LeakyReLU()

        self.layers = nn.Sequential(*layers)
        self.layer_input_shapes = self.get_layer_input_shapes()


class Conv_Net(SequentialModel):
    def __init__(
        self,
        width=64,
        dims=[(3, 32, 32), 10],
        bias=False,
        activation=nn.ReLU(),
    ):
        super(Conv_Net, self).__init__()

        self.dims = dims

        kernel_size = 3
        stride = 1
        padding = 1

        # determine the kernel size for the pooling layers for different
        # datasets to achieve a flat vector in the end
        if dims[0][-1] == 32:  # CIFAR10 case
            pooling_kernels = [1, 2, 2, 8]
        if dims[0][-1] == 28:  # MNIST case
            pooling_kernels = [1, 2, 2, 7]

        self.layers = nn.Sequential(
            nn.Conv2d(dims[0][0], width, kernel_size,
                      stride=stride, padding=padding, bias=bias),
            activation,
            nn.MaxPool2d(pooling_kernels[0]),
            ###
            nn.Conv2d(width, width * 2, kernel_size,
                      stride=stride, padding=padding, bias=bias),
            activation,
            nn.MaxPool2d(pooling_kernels[1]),
            ###
            nn.Conv2d(width * 2, width * 4, kernel_size,
                      stride=stride, padding=padding, bias=bias),
            activation,
            nn.MaxPool2d(pooling_kernels[2]),
            ###
            nn.Conv2d(width * 4, width * 8, kernel_size,
                      stride=stride, padding=padding, bias=bias),
            activation,
            nn.MaxPool2d(pooling_kernels[3]),
            ###
            nn.Flatten(),
            nn.Linear(width * 8, dims[-1], bias=bias),
        )

        self.layer_input_shapes = self.get_layer_input_shapes()


class ResNet(SequentialModel):
    """ResNet tuned to work on ImageNet and CIFAR10."""

    def __init__(
        self,
        width: int,
        dims: list = [[3, 224, 224], 1000],
        pretrained: bool = True,
        path_to_cifar_weights=Path("~/cifar_weights/"),
        **kwargs,
    ):
        super(ResNet, self).__init__(**kwargs)

        self.dims = dims

        if self.dims == [[3, 224, 224], 1000]:
            # ImageNet case
            if width == 18:
                weights = None
                if pretrained:
                    weights = tvm.ResNet18_Weights.IMAGENET1K_V1
                model = tvm.resnet18(weights=weights)
            elif width == 34:
                weights = None
                if pretrained:
                    weights = tvm.ResNet34_Weights.IMAGENET1K_V1
                model = tvm.resnet34(weights=weights)
            elif width == 50:
                weights = None
                if pretrained:
                    weights = tvm.ResNet50_Weights.IMAGENET1K_V1
                model = tvm.resnet50(weights=weights)
            elif width == 101:
                weights = None
                if pretrained:
                    weights = tvm.ResNet101_Weights.IMAGENET1K_V1
                model = tvm.resnet101(weights=weights)
            elif width == 152:
                weights = None
                if pretrained:
                    weights = tvm.ResNet152_Weights.IMAGENET1K_V1
                model = tvm.resnet152(weights=weights)
            else:
                raise NameError("This ResNet width is not supported.")

        elif self.dims == [[3, 32, 32], 10]:
            # CIFAR-10 case
            if width == 20:
                model = CIFAR10_ResNet(BasicBlock, [3, 3, 3])
                if pretrained:
                    model = load_cifar_weights(
                        model, path_to_cifar_weights, "resnet20-12fca82f.th")
            elif width == 32:
                model = CIFAR10_ResNet(BasicBlock, [5, 5, 5])
                if pretrained:
                    model = load_cifar_weights(
                        model, path_to_cifar_weights, "resnet32-d509ac18.th")
            elif width == 44:
                model = CIFAR10_ResNet(BasicBlock, [7, 7, 7])
                if pretrained:
                    model = load_cifar_weights(
                        model, path_to_cifar_weights, "resnet44-014dd654.th")
            elif width == 56:
                model = CIFAR10_ResNet(BasicBlock, [9, 9, 9])
                if pretrained:
                    model = load_cifar_weights(
                        model, path_to_cifar_weights, "resnet56-4bfd9763.th")
            elif width == 110:
                model = CIFAR10_ResNet(BasicBlock, [18, 18, 18])
                if pretrained:
                    model = load_cifar_weights(
                        model, path_to_cifar_weights, "resnet110-1d1ed7c2.th"
                    )
            elif width == 1202:
                model = CIFAR10_ResNet(BasicBlock, [200, 200, 200])
                if pretrained:
                    model = load_cifar_weights(
                        model, path_to_cifar_weights, "resnet1202-f3b1deed.th"
                    )
            else:
                raise NameError("This ResNet width is not supported.")

        else:
            raise ValueError(
                f"No ResNet implementation found for dimensions {dims}")

        layers = list(model._modules.values())
        # insert the flatten layer before FC so that the sequential wrapper
        # works
        layers.insert(-1, nn.Flatten())
        self.layers = nn.Sequential(*layers)
        self.layer_input_shapes = self.get_layer_input_shapes()


def load_cifar_weights(model, path, weights_name):
    weights_ = torch.load(
        path / weights_name,
        map_location=torch.device("cpu"),
    )["state_dict"]
    # remove "module." prefix
    weights = {}
    for k, v in weights_.items():
        k = ".".join(k.split(".")[1:])
        weights[k] = v

    weigths = OrderedDict(weights)
    model.load_state_dict(weights)

    return model


# Code below is based on the implementation of Yerlan Idelbayev. Models'
# binaries can be also found in the same repository.
# https://github.com/akamaster/pytorch_resnet_cifar10
# Copyright (c) 2018, Yerlan Idelbayev

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2], (0, 0, 0, 0, planes //
                                            4, planes // 4), "constant", 0
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes, self.expansion * planes, kernel_size=1,
                        stride=stride, bias=False
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CIFAR10_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(CIFAR10_ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
