import argparse
import torch
import os


# noinspection PyPep8
class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.conf = None

        # 图像文件的path
        self.parser.add_argument('--img_name', default='image1', help='目标图像文件名')
        self.parser.add_argument('--input_image_path', default=os.path.dirname(__file__) + '/training_data/input.png', help='指定图像的路径')
        self.parser.add_argument('--output_dir_path', default=os.path.dirname(__file__) + '/results', help='结果存储路径')

        # 尺寸相关
        self.parser.add_argument('--input_crop_size', type=int, default=64, help='生成器裁剪尺寸')
        self.parser.add_argument('--scale_factor', type=float, default=0.5, help='下采样尺寸因子')
        self.parser.add_argument('--X4', action='store_true', help='预期超分辨倍数（是否放大4倍）')

        # Network architecture
        self.parser.add_argument('--G_chan', type=int, default=64, help='# 生成器隐藏层通道数')
        self.parser.add_argument('--D_chan', type=int, default=64, help='# 鉴别器隐藏层通道数')
        self.parser.add_argument('--G_kernel_size', type=int, default=13, help='生成器估计核大小')
        self.parser.add_argument('--D_n_layers', type=int, default=7, help='鉴别器网络深度')
        self.parser.add_argument('--D_kernel_size', type=int, default=7, help='鉴别器网络卷积核大小')

        # 训练轮数
        self.parser.add_argument('--max_iters', type=int, default=3000, help='训练轮数')

        # Optimization hyper-parameters
        self.parser.add_argument('--g_lr', type=float, default=2e-4, help='生成器初始学习率')
        self.parser.add_argument('--d_lr', type=float, default=2e-4, help='鉴别器初始学习率')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='Adam momentum')

        # GPU
        self.parser.add_argument('--gpu_id', type=int, default=0, help='gpu id number')

        # Kernel post processing
        self.parser.add_argument('--n_filtering', type=float, default=40, help='过滤卷积核中的过小值')

        # ZSSR configuration
        self.parser.add_argument('--do_ZSSR', action='store_true', help='是否启用超分辨--即ZSSR')
        self.parser.add_argument('--noise_scale', type=float, default=1., help='ZSSR噪声尺寸，具体见ZSSR代码')
        self.parser.add_argument('--real_image', action='store_true', help='for real images')

    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)
        self.set_gpu_device()
        self.clean_file_name()
        self.set_output_directory()
        self.conf.G_structure = [7, 5, 3, 1, 1, 1]
        print("Scale Factor: %s \tZSSR: %s \tReal Image: %s" % (('X4' if self.conf.X4 else 'X2'), str(self.conf.do_ZSSR), str(self.conf.real_image)))
        return self.conf

    def clean_file_name(self):
        """提取出纯净的文件名以进行保存"""
        self.conf.img_name = self.conf.input_image_path.split('\\')[-1].replace('ZSSR', '') \
            .replace('real', '').replace('__', '').split('_.')[0].split('.')[0]

    def set_gpu_device(self):
        """设置GPU（如果有）"""
        if os.environ.get('CUDA_VISIBLE_DEVICES', '') == '':
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.conf.gpu_id)
            torch.cuda.set_device(0)
        else:
            torch.cuda.set_device(self.conf.gpu_id)

    def set_output_directory(self):
        """定义结果文件夹名，并创建"""
        self.conf.output_dir_path = os.path.join(self.conf.output_dir_path, self.conf.img_name)
        # 如果指定的文件夹名存在，则在原文件夹名后加一个“l”（这里想用时间戳，但是懒得改，先这样）
        while os.path.isdir(self.conf.output_dir_path):
            self.conf.output_dir_path += 'l'
        os.makedirs(self.conf.output_dir_path)
