# coding = UTF-8


import numpy as np
from torch.utils.data import Dataset
from imresize import imresize
from util import read_image, create_gradient_map, im2tensor, create_probability_map, nn_interpolation


class DataGenerator(Dataset):
    """
    生成器加载一次图像，在初始化时计算它的梯度图，然后在调用时输出该图像的裁剪版本
    """

    def __init__(self, conf, gan):
        # 默认的Shape
        self.g_input_shape = conf.input_crop_size
        self.d_input_shape = gan.G.output_size  # 由G生成的输入D的图片大小
        self.d_output_shape = self.d_input_shape - gan.D.forward_shave

        # 读取输入图像
        self.input_image = read_image(conf.input_image_path) / 255.
        self.shave_edges(scale_factor=conf.scale_factor, real_image=conf.real_image)

        self.in_rows, self.in_cols = self.input_image.shape[0:2]

        # 为选中的裁剪图像创建概率图
        self.crop_indices_for_g, self.crop_indices_for_d = self.make_list_of_crop_indices(conf=conf)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        """为生成器和鉴别器获取裁剪图像 """
        g_in = self.next_crop(for_g=True, idx=idx)
        d_in = self.next_crop(for_g=False, idx=idx)

        return g_in, d_in

    def next_crop(self, for_g, idx):
        """根据预定的索引列表返回一个裁剪图像，D的返回值会添加噪声"""
        size = self.g_input_shape if for_g else self.d_input_shape
        top, left = self.get_top_left(size, for_g, idx)
        crop_im = self.input_image[top:top + size, left:left + size, :]
        if not for_g:  # 为D的返回图像添加噪声
            crop_im += np.random.randn(*crop_im.shape) / 255.0
        return im2tensor(crop_im)

    def make_list_of_crop_indices(self, conf):
        iterations = conf.max_iters
        prob_map_big, prob_map_sml = self.create_prob_maps(scale_factor=conf.scale_factor)
        crop_indices_for_g = np.random.choice(a=len(prob_map_sml), size=iterations, p=prob_map_sml)
        crop_indices_for_d = np.random.choice(a=len(prob_map_big), size=iterations, p=prob_map_big)
        return crop_indices_for_g, crop_indices_for_d

    def create_prob_maps(self, scale_factor):
        # 为输入图像和缩小图像创建损失图
        loss_map_big = create_gradient_map(self.input_image)
        loss_map_sml = create_gradient_map(imresize(im=self.input_image, scale_factor=scale_factor, kernel='cubic'))
        # 创建相应的概率图
        prob_map_big = create_probability_map(loss_map_big, self.d_input_shape)
        prob_map_sml = create_probability_map(nn_interpolation(loss_map_sml, int(1 / scale_factor)), self.g_input_shape)
        return prob_map_big, prob_map_sml

    def shave_edges(self, scale_factor, real_image):
        """去除边缘的像素，防止Bug"""
        # 裁剪 10 个像素以避免生成样本中的边界效应
        if not real_image:
            self.input_image = self.input_image[10:-10, 10:-10, :]
        # 裁剪部分像素，使得图像尺寸可以被比例因子整除
        # 例如601x601不能被4整除，变成600x600
        sf = int(1 / scale_factor)
        shape = self.input_image.shape
        self.input_image = self.input_image[:-(shape[0] % sf), :, :] if shape[0] % sf > 0 else self.input_image
        self.input_image = self.input_image[:, :-(shape[1] % sf), :] if shape[1] % sf > 0 else self.input_image

    def get_top_left(self, size, for_g, idx):
        """将裁剪图像的中心索引转换成其左上角的索引"""
        center = self.crop_indices_for_g[idx] if for_g else self.crop_indices_for_d[idx]
        row, col = int(center / self.in_cols), center % self.in_cols
        top, left = min(max(0, row - size // 2), self.in_rows - size), min(max(0, col - size // 2), self.in_cols - size)
        # 选择偶数索引（以避免与损失图 for_g 不一致）
        return top - top % 2, left - left % 2