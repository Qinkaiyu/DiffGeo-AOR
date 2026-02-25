import os
import os.path as osp
import random
from collections import Counter
import nibabel as nib
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from scipy import ndimage
import pandas as pd
import torch
def add_salt_peper_3D(image, amout):
    s_vs_p = 0.5
    noisy_img = np.copy(image)
    num_salt = np.ceil(amout * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_img[coords[0], coords[1]] = 1.
    num_pepper = np.ceil(amout * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_img[coords[0], coords[1]] = 0.
    return noisy_img
def add_salt_peper(image, amout):
    s_vs_p = 0.5
    noisy_img = np.copy(image)

    num_salt = np.ceil(amout * image.shape[0] * image.shape[1] * s_vs_p)

    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_img[coords[0], coords[1], :] = 1.

    num_pepper = np.ceil(amout * image.shape[0] * image.shape[1] * (1. - s_vs_p))

    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_img[coords[0], coords[1], :] = 0.

    return noisy_img
def scale_image(image, patch_size):
    image = cv2.resize(image, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
    return image


def resize_oct_data_trans(data, size):
    """
    Resize the data to the input size
    """
    input_D, input_H, input_W = size[0], size[1], size[2]
    data = data.squeeze()
    [depth, height, width] = data.shape
    scale = [input_D * 1.0 / depth, input_H * 1.0 / height, input_W * 1.0 / width]
    data = ndimage.interpolation.zoom(data, scale, order=0)
    # data = data.unsqueeze()
    return data
# class ExcelDataset(Dataset):
#     """
#     从Excel(CSV)文件读取数据的Dataset类。
#     会读取'image_file_path'列作为图片路径，'pathology'列作为标签。
#     """
#     def __init__(self, excel_file, images_root=None, transforms=None):
#         """
#         Args:
#             excel_file (str): CSV文件路径。
#             images_root (str): 图片根目录路径，如果为None则使用image列中的绝对路径。
#             transforms: 应用于图片的变换。
#         """
#         self.excel_file = excel_file
#         self.images_root = images_root
#         self.transforms = transforms
        
#         # 读取CSV文件
#         self.data = pd.read_csv(excel_file)
        
#         # 检查必需的列是否存在
#         if 'image_file_path' not in self.data.columns:
#             raise ValueError("CSV文件必须包含 'image_file_path' 列")
#         if 'pathology' not in self.data.columns:
#             raise ValueError("CSV文件必须包含 'pathology' 列")
        
#         # --- 修改点 1: 读取 image_file_path 和 pathology 列 ---
#         self.image_paths = self.data['image_file_path'].tolist()
#         # 先读取原始的字符串标签
#         original_labels = self.data['pathology']

#         # --- 修改点 2: 将字符串标签映射为数字 ---
#         # 创建一个从字符串到整数的映射关系
#         # 您可以根据需要调整这个映射
#         self.label_map = {'BENIGN_WITHOUT_CALLBACK': 0, 'BENIGN': 1, 'MALIGNANT': 2}
        
#         # 使用 .map() 函数应用这个映射，并将结果转换为列表
#         # 对于任何不在 label_map 中的值 (例如 NaN 或其他字符串)，.map 会产生 NaT
#         # 使用 .fillna(-1).astype(int) 可以将这些无效标签设为-1，便于调试
#         self.labels = original_labels.map(self.label_map).fillna(-1).astype(int).tolist()

#         # 打印初始化信息
#         print(f"ExcelDataset初始化: 从 {excel_file} 读取数据")
#         print(f"标签映射关系: {self.label_map}")
        
#         # 打印原始字符串标签的分布情况，便于检查
#         print("原始标签分布:")
#         print(dict(sorted(Counter(original_labels).items())))
        
#         print(f"数据集大小: {len(self.labels)}")
#         # 确保所有标签都成功转换了
#         if -1 in self.labels:
#             print("警告: 数据集中存在无法识别的标签，已被标记为 -1")
        
#         print(f"数字化后标签范围: {min(self.labels)} - {max(self.labels)}")
        
#     def __getitem__(self, index):
#         # 获取图片路径和（数字化后的）标签
#         img_path = self.image_paths[index]
#         label = self.labels[index]
        
#         # 构建完整的图片路径
#         if self.images_root:
#             # 如果images_root不为None，则拼接路径
#             full_path = os.path.join(self.images_root, img_path)
#         else:
#             # 否则，假设img_path已经是完整路径
#             full_path = img_path
            
#         # 加载图片
#         try:
#             img = Image.open(full_path)
#             # 如果是单通道（灰度图），转换为RGB
#             if img.mode != "RGB":
#                 img = img.convert("RGB")
#         except Exception as e:
#             print(f"加载图片失败: {full_path}, 错误: {e}")
#             # 如果加载失败，返回一个黑色的默认图片和一个无效标签
#             img = Image.new('RGB', (224, 224), color='black')
#             # 也可以选择让程序在这里崩溃，以强制修复数据问题
#             # raise e
            
#         # 应用图片变换
#         if self.transforms:
#             img = self.transforms(img)
            
#         return img, label
    
#     def __len__(self):
#         return len(self.labels)
    
#     def get_label_distribution(self):
#         """返回数字化后标签的分布信息"""
#         label_counts = Counter(self.labels)
#         # 将数字标签转换回原始的字符串，让输出结果更易读
#         distribution = {
#             # 创建一个反向映射 {0: 'BENIGN', 1: 'MALIGNANT'}
#             reverse_map.get(label, 'UNKNOWN'): count 
#             for label, count in label_counts.items()
#             for reverse_map in [{v: k for k, v in self.label_map.items()}]
#         }
#         return dict(sorted(distribution.items()))
class ExcelDataset(Dataset):
    """
    从Excel文件读取数据的Dataset类
    Excel文件需要包含以下列：
    - image: 图片文件路径
    - quality: 标签值
    """
    def __init__(self, excel_file, images_root=None, transforms=None):
        """
        Args:
            excel_file (str): Excel文件路径
            images_root (str): 图片根目录路径，如果为None则使用image列中的绝对路径
            transforms: 图片变换操作
        """
        self.excel_file = excel_file
        self.images_root = images_root
        self.transforms = transforms
        
        # 读取Excel文件
        self.data = pd.read_csv(excel_file)
        
        # 检查必需的列是否存在
        if 'image' not in self.data.columns:
            raise ValueError("Excel文件必须包含'image'列")
        if 'quality' not in self.data.columns:
            raise ValueError("Excel文件必须包含'quality'列")
        
        # 获取图片路径和标签
        self.image_paths = self.data['image'].tolist()
        self.labels = self.data['quality'].tolist()
        
        print(f"ExcelDataset初始化: 从{excel_file}读取数据")
        print(f"数据集大小: {len(self.labels)}")
        print(f"标签范围: {min(self.labels)} - {max(self.labels)}")
        
    def __getitem__(self, index):
        # 获取图片路径和标签
        img_path = self.image_paths[index]
        label = self.labels[index]
        
        # 构建完整的图片路径
        if self.images_root:
            full_path = os.path.join(self.images_root, img_path)
        else:
            full_path = img_path
            
        # 加载图片
        try:
            img = Image.open(full_path)
            if img.mode == "L":
                img = img.convert("RGB")
        except Exception as e:
            print(f"加载图片失败: {full_path}, 错误: {e}")
            # 创建一个默认的RGB图片
            img = Image.new('RGB', (224, 224), color='black')
        
        # 应用图片变换
        if self.transforms:
            img = self.transforms(img)
            
        return img, label
    
    def __len__(self):
        return len(self.labels)
    
    def get_label_distribution(self):
        """返回标签分布信息"""
        from collections import Counter
        label_counts = Counter(self.labels)
        return dict(sorted(label_counts.items()))
class RegressionDataset(Dataset):
    def __init__(self, images_root, data_file, transforms):
        self.images_root = images_root
        self.labels = []
        self.images_file = []
        self.transforms = transforms
        with open(data_file) as fin:
            for line in fin:
                # image_file, image_label = line.split()
                splits = line.split()
                image_file = splits[0]
                labels = splits[1:]
                self.labels.append([int(label) for label in labels])
                self.images_file.append(image_file)
        self.name = osp.splitext(osp.basename(data_file))[0].lower()
        if "val" in self.name or "test" in self.name:
            print(f"Dataset prepare: val/test data_file: {data_file}")
        elif "train" in self.name:
            print(f"Dataset prepare: train data_file: {data_file}")
        else:
            raise ValueError(f"Invalid data_file: {data_file}")
        print(f"Dataset prepare: len of labels: {len(self.labels[0])}")
        print(f"Dataset prepare: len of dataset: {len(self.labels)}")
    def __getitem__(self, index):
        img_file, target_list = self.images_file[index], self.labels[index]
        if "val" in self.name or "test" in self.name:
            target = target_list[len(target_list) // 2]
        else:
            target = random.choice(target_list)

        full_file = os.path.join(self.images_root, img_file)
        img = Image.open(full_file)
        if img.mode == "L":
            img = img.convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return img, target
    def __len__(self):
        return len(self.labels)
class GAMMA_dataset(Dataset):
    def __init__(self,
                 dataset_root,
                 oct_img_size,
                 fundus_img_size,
                 mode='train',
                 label_file='',
                 filelists=None,
                 ):


        self.dataset_root = dataset_root
        self.input_D = oct_img_size[0][0]
        self.input_H = oct_img_size[0][1]
        self.input_W = oct_img_size[0][2]

        self.fundus_train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomApply([
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            # normalize,
        ])

        self.oct_train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
        ])
        self.fundus_val_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.oct_val_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.mode = mode.lower()
        label = {row['data']: row[1:].values
                 for _, row in pd.read_excel(label_file).iterrows()}
        self.file_list = []
        for f in filelists:
            filename = os.path.basename(f)  # 提取文件名
            if filename.isdigit():  # 确保文件名是纯数字
                self.file_list.append([f, label[int(filename)]])
    def __getitem__(self, idx):
        data = dict()
        real_index, label = self.file_list[idx]
        # fundus_img_path = os.path.join(self.dataset_root.replace('/MGamma/', '/multi-modality_images/'), real_index,
        #                                real_index + ".jpg")
        # fundus_img_path = os.path.join(self.dataset_root.replace('/MGamma/', '/multi-modality_images/'), real_index,
        #                                real_index + ".png")

        oct_nii = nib.load(os.path.join(self.dataset_root, real_index, f'processed_data_{real_index}.nii'))
        oct_img = oct_nii.get_fdata()
        # fundus_img = scale_image(fundus_img, 336)
        oct_img = resize_oct_data_trans(oct_img, (96, 96, 96))

        # oct_img = oct_img / 255.0
        # fundus_img = fundus_img / 255.0
        if self.mode == "train":
            oct_img = self.oct_train_transforms(oct_img.astype(np.float32))
        else:
            oct_img = self.oct_val_transforms(oct_img)
        
        # 将1通道OCT图像转换为3通道（复制通道）
        oct_img = oct_img.repeat(3, 1, 1, 1)  # 从 [1, D, H, W] 变为 [3, D, H, W]
        
        label = label.argmax()
        return oct_img, label

    def __len__(self):
        return len(self.file_list)

    def __resize_oct_data__(self, data):
        """
        Resize the data to the input size
        """
        data = data.squeeze()
        [depth, height, width] = data.shape
        scale = [self.input_D * 1.0 / depth, self.input_H * 1.0 / height, self.input_W * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=0)
        # data = data.unsqueeze()
        return data