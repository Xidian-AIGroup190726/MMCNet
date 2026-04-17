
from torch.utils.data import Dataset
import torch
from libtiff import TIFF
import numpy as np
import cv2


def data_process(pan_path, ms_path, train_label_path, test_label_path, Ms4_patch_size):

    # 读取图片、标签
    pan_tif = TIFF.open(pan_path, mode='r')
    pan_np = pan_tif.read_image()
    print('原始pan图的形状;', np.shape(pan_np))

    ms4_tif = TIFF.open(ms_path, mode='r')
    ms4_np = ms4_tif.read_image()
    print('原始ms4图的形状：', np.shape(ms4_np))

    train_label_np = np.load(train_label_path)
    test_label_np = np.load(test_label_path)

    # ms4与pan图补零
    Interpolation = cv2.BORDER_REFLECT_101
    # cv2.BORDER_REPLICATE： 进行复制的补零操作;
    # cv2.BORDER_REFLECT:  进行翻转的补零操作:gfedcba|abcdefgh|hgfedcb;
    # cv2.BORDER_REFLECT_101： 进行翻转的补零操作:gfedcb|abcdefgh|gfedcb;
    # cv2.BORDER_WRAP: 进行上下边缘调换的外包复制操作:bcdegh|abcdefgh|abcdefg;

    top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),
                                                    int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))
    ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)
    print('补零后的ms4图的形状：', np.shape(ms4_np))

    Pan_patch_size = Ms4_patch_size * 4  # pan截块的边长
    top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2),
                                                    int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2))
    pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)
    print('补零后的pan图的形状：', np.shape(pan_np))

    gt_np = train_label_np - 1  # 通过减一后将类别归到 0-N，而未标注类标签变为255

    label_element, element_count = np.unique(gt_np, return_counts=True)
    Categories_Number = len(label_element) - 1  # 数据的类别数 9-1=8

    print('训练集类标：', label_element)
    print('训练集各类样本数：', element_count)
    print('训练标注的类别数：', Categories_Number)

    test_gt_np = test_label_np - 1  # 通过减一后将类别归到 0-N，而未标注类标签变为255

    test_label_element, test_element_count = np.unique(test_gt_np, return_counts=True)
    test_Categories_Number = len(test_label_element) - 1

    print('测试类标：', test_label_element)
    print('测试集各类样本数：', test_element_count)
    print('测试标注的类别数：', test_Categories_Number)

    # 统计 label 图中的 row、column 所有标签个数
    label_row, label_column = np.shape(gt_np)  # 获取 label 图的行、列
    ground_xy = np.array([[]] * Categories_Number).tolist()
    ground_xy_allData = np.arange(label_row * label_column * 2)
    ground_xy_allData = ground_xy_allData.reshape(label_row * label_column, 2)

    count = 0
    for row in range(label_row):
        for column in range(label_column):
            ground_xy_allData[count] = [row, column]
            count = count + 1
            if gt_np[row][column] != 255:
                ground_xy[int(gt_np[row][column])].append([row, column])

    # 统计 测试label 图中的 row、column 所有标签个数
    test_label_row, test_label_column = np.shape(test_gt_np)  # 获取 label 图的行、列
    test_ground_xy = np.array([[]] * test_Categories_Number).tolist()
    test_ground_xy_allData = np.arange(test_label_row * test_label_column * 2)  # range
    test_ground_xy_allData = test_ground_xy_allData.reshape(test_label_row * test_label_column, 2)

    test_count = 0
    for test_row in range(test_label_row):
        for test_column in range(test_label_column):
            test_ground_xy_allData[test_count] = [test_row, test_column]
            test_count = test_count + 1

            if test_gt_np[test_row][test_column] != 255:
                test_ground_xy[int(test_gt_np[test_row][test_column])].append([test_row, test_column])

    # 标签内打乱
    for categories in range(Categories_Number):  # 8类：从第0类开始循环，打乱标签
        ground_xy[categories] = np.array(ground_xy[categories])
        shuffle_array = np.arange(0, len(ground_xy[categories]), 1)  # 三个参数 np.arange(a, b, c): 起点a，终点b，步长c
        np.random.shuffle(shuffle_array)
        ground_xy[categories] = ground_xy[categories][shuffle_array]  # 类别索引 + 位置

    shuffle_array = np.arange(0, label_row * label_column, 1)
    np.random.shuffle(shuffle_array)
    ground_xy_allData = ground_xy_allData[shuffle_array]

    # 测试标签内打乱
    for test_categories in range(test_Categories_Number):  # 8类：从第0类开始循环，打乱标签
        test_ground_xy[test_categories] = np.array(test_ground_xy[test_categories])
        test_shuffle_array = np.arange(0, len(test_ground_xy[test_categories]),
                                       1)  # 三个参数 np.arange(a, b, c): 起点a，终点b，步长c
        np.random.shuffle(test_shuffle_array)
        test_ground_xy[test_categories] = test_ground_xy[test_categories][test_shuffle_array]  # 类别索引 + 位置

    test_shuffle_array = np.arange(0, test_label_row * test_label_column, 1)
    np.random.shuffle(test_shuffle_array)
    test_ground_xy_allData = test_ground_xy_allData[test_shuffle_array]  # shape:(820148*2, ), size:1640296

    ######      从训练、测试的label中 按比例 选择数据      #####
    ######                  进行训练                   #####
    ground_xy_train = []
    ground_xy_test = []
    label_train = []
    label_test = []

    for categories in range(Categories_Number):
        categories_number = len(ground_xy[categories])
        for i in range(categories_number):
            ground_xy_train.append(ground_xy[categories][i])

        label_train = label_train + [categories for x in range(int(categories_number))]

    for test_categories in range(test_Categories_Number):
        test_categories_number = len(test_ground_xy[test_categories])
        # print('aaa', categories_number)
        for i in range(test_categories_number):
            ground_xy_test.append(test_ground_xy[test_categories][i])

        label_test = label_test + [test_categories for x in range(int(test_categories_number))]

    label_train = np.array(label_train)  # 训练集label图片位置
    label_test = np.array(label_test)
    ground_xy_train = np.array(ground_xy_train)  # 训练集的图片位置
    ground_xy_test = np.array(ground_xy_test)

    ######         训练数据、测试数据数据集内打乱         ####
    shuffle_array = np.arange(0, len(label_test), 1)
    np.random.shuffle(shuffle_array)
    label_test = label_test[shuffle_array]
    ground_xy_test = ground_xy_test[shuffle_array]

    shuffle_array = np.arange(0, len(label_train), 1)
    np.random.shuffle(shuffle_array)
    label_train = label_train[shuffle_array]
    ground_xy_train = ground_xy_train[shuffle_array]

    label_train = torch.from_numpy(label_train).type(torch.LongTensor)
    label_test = torch.from_numpy(label_test).type(torch.LongTensor)
    ground_xy_train = torch.from_numpy(ground_xy_train).type(torch.LongTensor)
    ground_xy_test = torch.from_numpy(ground_xy_test).type(torch.LongTensor)
    ground_xy_allData = torch.from_numpy(ground_xy_allData).type(torch.LongTensor)

    print('训练样本数：', len(label_train))
    print('测试样本数：', len(label_test))

    # 数据归一化
    ms4 = min_max_norm(ms4_np)
    pan = min_max_norm(pan_np)

    ms4 = np.array(ms4).transpose((2, 0, 1))  # 调整通道
    pan = np.expand_dims(pan, axis=0)  # 二维数据进网络前要加一维

    ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
    pan = torch.from_numpy(pan).type(torch.FloatTensor)

    train_data = MyData(ms4, pan, label_train, ground_xy_train, Ms4_patch_size)
    test_data = MyData(ms4, pan, label_test, ground_xy_test, Ms4_patch_size)
    all_data = MyData1(ms4, pan, ground_xy_allData, Ms4_patch_size)

    return train_data, test_data, all_data, label_test


def min_max_norm(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image


class MyData(Dataset):

    def __init__(self, MS4, Pan, Label, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan
        self.train_labels = Label
        self.gt_xy = xy

        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)  # 计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)

        # torch.Size([4, 16, 16])
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size, y_ms:y_ms + self.cut_ms_size]

        # torch.Size([1, 64, 64])
        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size, y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        target = self.train_labels[index]

        return image_ms, image_pan, target, locate_xy

    def __len__(self):
        return len(self.gt_xy)


class MyData1(Dataset):
    def __init__(self, MS4, Pan, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan

        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)  # 计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)

        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size, y_ms:y_ms + self.cut_ms_size]
        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size, y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        # return image_ms, image_pan, image_mshpan, locate_xy
        return image_ms, image_pan, locate_xy

    def __len__(self):
        return len(self.gt_xy)
