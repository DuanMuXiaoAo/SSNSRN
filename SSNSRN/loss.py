import torch
import torch.nn as nn
from torch.autograd import Variable
from util import shave_a2b, resize_tensor_w_kernel, create_penalty_mask, map2tensor


# noinspection PyUnresolvedReferences
class GANLoss(nn.Module):
    """D outputs a [0,1] map of size of the input. This map is compared in a pixel-wise manner to 1/0 according to
    whether the input is real (i.e. from the input image) or fake (i.e. from the Generator)"""

    def __init__(self, d_last_layer_size):
        super(GANLoss, self).__init__()
        # The loss function is applied after the pixel-wise comparison to the true label (0/1)
        self.loss = nn.L1Loss(reduction='mean')
        # Make a shape
       # d_last_layer_shape = [1, 1, d_last_layer_size, d_last_layer_size]
        d_last_layer_shape = [d_last_layer_size]
        # The two possible label maps are pre-prepared
        self.label_tensor_fake = Variable(torch.zeros(d_last_layer_shape).cuda(), requires_grad=False)
        self.label_tensor_real = Variable(torch.ones(d_last_layer_shape).cuda(), requires_grad=False)

    def forward(self, d_last_layer, is_d_input_real):
        # Determine label map according to whether current input to discriminator is real or fake
        label_tensor = self.label_tensor_real if is_d_input_real else self.label_tensor_fake
        # Compute the loss
        return self.loss(d_last_layer, label_tensor)


class DownScaleLoss(nn.Module):
    """计算当前核和理想核之间的MSE"""

    def __init__(self, scale_factor):
        super(DownScaleLoss, self).__init__()
        self.loss = nn.MSELoss()

        # 这里是默认的bicubic核
        bicubic_k = [[0.0001373291015625, 0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375, 0.0004119873046875, 0.0001373291015625],
                     [0.0004119873046875, 0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125, 0.0012359619140625, 0.0004119873046875],
                     [-.0013275146484375, -0.0039825439453130, 0.0128326416015625, 0.0491180419921875, 0.0491180419921875, 0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                     [-.0050811767578125, -0.0152435302734375, 0.0491180419921875, 0.1880035400390630, 0.1880035400390630, 0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                     [-.0050811767578125, -0.0152435302734375, 0.0491180419921875, 0.1880035400390630, 0.1880035400390630, 0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                     [-.0013275146484380, -0.0039825439453125, 0.0128326416015625, 0.0491180419921875, 0.0491180419921875, 0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                     [0.0004119873046875, 0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125, 0.0012359619140625, 0.0004119873046875],
                     [0.0001373291015625, 0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375, 0.0004119873046875, 0.0001373291015625]]

        """
        # 这里是生成的云图理想核，改成了8x8的大小
        bicubic_k = [
        [9.2917478e-5, 0.0002802391, 0.00033742454, 0.00032558962, 0.00035733898, 0.00049565471, 0.00056746864, 0.000142089, 4.0359846e-5],
        [0.0002322564, 0.00055141386, 0.00090239767, 0.0014666776, 0.0022270374, 0.002379105, 0.0014578431, 0.00046559583, 0.00015521224],
        [0.00021016209, 0.00076094549, 0.0033620032, 0.0095453365, 0.015302635, 0.01504796, 0.0088010067, 0.0025186741, 0.00033980212],
        [0.00052359956, 0.0019261058, 0.01106353, 0.027906466, 0.041903008, 0.041448634, 0.026884394, 0.009749122, 0.0012255679],
        [0.00038139708, 0.0035021009, 0.018456757, 0.043954231, 0.065173879, 0.065296859, 0.044191401, 0.018187635, 0.002997726],
        [0.00016453787, 0.0031629964, 0.018383939, 0.044014581, 0.065487616, 0.066534482, 0.045491647, 0.018940872, 0.0028807994],
        [0.00011377558, 0.001497703, 0.010417797, 0.027381649, 0.041937605, 0.042893488, 0.028910307, 0.011073621, 0.0015136714],
        [0.00016826532, 0.00089228712, 0.0030453769, 0.0089636473, 0.015113635, 0.015272978, 0.0097867688, 0.0034841134, 0.00093906],
        [0.00013363639, 0.00040883711, 0.0005672542, 0.0010611124, 0.0019276642, 0.0019404214, 0.0013666914, 0.00084756344, 0.00057969178]]
        """
        self.bicubic_kernel = Variable(torch.Tensor(bicubic_k).cuda(), requires_grad=False)
        self.scale_factor = scale_factor

    def forward(self, g_input, g_output):
        downscaled = resize_tensor_w_kernel(im_t=g_input, k=self.bicubic_kernel, sf=self.scale_factor)
        # Shave the downscaled to fit g_output
        return self.loss(g_output, shave_a2b(downscaled, g_output))


class SumOfWeightsLoss(nn.Module):
    """ Encourages the kernel G is imitating to sum to 1 """

    def __init__(self):
        super(SumOfWeightsLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(torch.ones(1).to(kernel.device), torch.sum(kernel))


class CentralizedLoss(nn.Module):
    """ Penalizes distance of center of mass from K's center"""

    def __init__(self, k_size, scale_factor=.5):
        super(CentralizedLoss, self).__init__()
        self.indices = Variable(torch.arange(0., float(k_size)).cuda(), requires_grad=False)
        wanted_center_of_mass = k_size // 2 + 0.5 * (int(1 / scale_factor) - k_size % 2)
        self.center = Variable(torch.FloatTensor([wanted_center_of_mass, wanted_center_of_mass]).cuda(), requires_grad=False)
        self.loss = nn.MSELoss()

    def forward(self, kernel):
        """Return the loss over the distance of center of mass from kernel center """
        r_sum, c_sum = torch.sum(kernel, dim=1).reshape(1, -1), torch.sum(kernel, dim=0).reshape(1, -1)
        return self.loss(torch.stack((torch.matmul(r_sum, self.indices) / torch.sum(kernel),
                                      torch.matmul(c_sum, self.indices) / torch.sum(kernel))), self.center)


class BoundariesLoss(nn.Module):
    """ Encourages sparsity of the boundaries by penalizing non-zeros far from the center """

    def __init__(self, k_size):
        super(BoundariesLoss, self).__init__()
        self.mask = map2tensor(create_penalty_mask(k_size, 30))
        self.zero_label = Variable(torch.zeros(k_size).cuda(), requires_grad=False)
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(kernel * self.mask, self.zero_label)


class SparsityLoss(nn.Module):
    """ Penalizes small values to encourage sparsity """
    def __init__(self):
        super(SparsityLoss, self).__init__()
        self.power = 0.2
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(torch.abs(kernel) ** self.power, torch.zeros_like(kernel))
