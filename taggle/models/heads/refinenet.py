import torch.nn as nn


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class MultiResolutionFusion(nn.Module):
    def __init__(self, out_feats, *shapes):
        super().__init__()

        _, max_size = max(shapes, key=lambda x: x[1])

        self.scale_factors = []
        for i, shape in enumerate(shapes):
            feat, size = shape
            if max_size % size != 0:
                raise ValueError("max_size not divisble by shape {}".format(i))

            self.scale_factors.append(max_size // size)
            self.add_module(
                "resolve{}".format(i),
                nn.Conv2d(
                    feat,
                    out_feats,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False))

    def forward(self, *xs):

        output = self.resolve0(xs[0])
        if self.scale_factors[0] != 1:
            output = nn.functional.interpolate(
                output,
                scale_factor=self.scale_factors[0],
                mode='bilinear',
                align_corners=True)

        for i, x in enumerate(xs[1:], 1):
            output += self.__getattr__("resolve{}".format(i))(x)
            if self.scale_factors[i] != 1:
                output = nn.functional.interpolate(
                    output,
                    scale_factor=self.scale_factors[i],
                    mode='bilinear',
                    align_corners=True)

        return output


class ChainedResidualPool(nn.Module):
    def __init__(self, feats):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 4):
            self.add_module(
                "block{}".format(i),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                    nn.Conv2d(
                        feats,
                        feats,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False)))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(1, 4):
            path = self.__getattr__("block{}".format(i))(path)
            x = x + path

        return x


class ChainedResidualPoolImproved(nn.Module):
    def __init__(self, feats):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 5):
            self.add_module(
                "block{}".format(i),
                nn.Sequential(
                    nn.Conv2d(
                        feats,
                        feats,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False),
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2)))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(1, 5):
            path = self.__getattr__("block{}".format(i))(path)
            x += path

        return x


class BaseRefineNetBlock(nn.Module):
    def __init__(self, features, residual_conv_unit, multi_resolution_fusion,
                 chained_residual_pool, *shapes):
        super().__init__()

        for i, shape in enumerate(shapes):
            feats = shape[0]
            self.add_module(
                "rcu{}".format(i),
                nn.Sequential(
                    residual_conv_unit(feats), residual_conv_unit(feats)))

        if len(shapes) != 1:
            self.mrf = multi_resolution_fusion(features, *shapes)
        else:
            self.mrf = None

        self.crp = chained_residual_pool(features)
        self.output_conv = residual_conv_unit(features)

    def forward(self, *xs):
        rcu_xs = []

        for i, x in enumerate(xs):
            rcu_xs.append(self.__getattr__("rcu{}".format(i))(x))

        if self.mrf is not None:
            out = self.mrf(*rcu_xs)
        else:
            out = rcu_xs[0]

        out = self.crp(out)
        return self.output_conv(out)


class RefineNetBlock(BaseRefineNetBlock):
    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit, MultiResolutionFusion,
                         ChainedResidualPool, *shapes)


class RefineNetBlockImprovedPooling(BaseRefineNetBlock):
    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit, MultiResolutionFusion,
                         ChainedResidualPoolImproved, *shapes)


class BaseRefineNetHead(nn.Module):
    def __init__(self,
                 input_shape,
                 encoder_channels,
                 refinenet_block,
                 num_class=1,
                 features=256,):
        super().__init__()
        input_size = input_shape

        if input_size % 32 != 0:
            raise ValueError("{} not divisble by 32".format(input_shape))
        self.layer1_rn = nn.Conv2d(
            encoder_channels[3], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(
            encoder_channels[2], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(
            encoder_channels[1], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(
            encoder_channels[0], 2 * features, kernel_size=3, stride=1, padding=1, bias=False)

        self.refinenet4 = refinenet_block(2 * features,
                                          (2 * features, input_size // 32))
        self.refinenet3 = refinenet_block(features,
                                          (2 * features, input_size // 32),
                                          (features, input_size // 16))
        self.refinenet2 = refinenet_block(features,
                                          (features, input_size // 16),
                                          (features, input_size // 8))
        self.refinenet1 = refinenet_block(features, (features, input_size // 8),
                                          (features, input_size // 4))
        self.output_conv = nn.Sequential(
            ResidualConvUnit(features), ResidualConvUnit(features),
            nn.Conv2d(
                features,
                num_class,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True))

    def forward(self, x):
        layer_1_rn = self.layer1_rn(x[3])
        layer_2_rn = self.layer2_rn(x[2])
        layer_3_rn = self.layer3_rn(x[1])
        layer_4_rn = self.layer4_rn(x[0])
        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)
        out = self.output_conv(path_1)
        out = nn.functional.interpolate(
            out, scale_factor=4, mode='bilinear', align_corners=True)
        return out


class RefineNetPoolingImproveHead(BaseRefineNetHead):
    __name__ = "RefineNetPoolingImproveHead"

    def __init__(self,
                 input_shape,
                 encoder_channels,
                 num_class=1,
                 features=256):
        """Multi-path 4-Cascaded RefineNet for image segmentation with improved pooling
        Args:
            input_shape ((int, int)): (channel, size) assumes input has
                equal height and width
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super().__init__(
            input_shape,
            encoder_channels,
            RefineNetBlockImprovedPooling,
            num_class=num_class,
            features=features)


class RefineNetHead(BaseRefineNetHead):
    __name__ = "RefineNetHead"

    def __init__(self,
                 input_shape,
                 encoder_channels,
                 num_class=1,
                 features=256):
        """Multi-path 4-Cascaded RefineNet for image segmentation
        Args:
            input_shape ((int, int)): (channel, size) assumes input has
                equal height and width
            refinenet_block (block): RefineNet Block
            num_class (int, optional): number of classes
            features (int, optional): number of features in refinenet
        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super().__init__(
            input_shape,
            encoder_channels,
            RefineNetBlock,
            num_class,
            features=features)
