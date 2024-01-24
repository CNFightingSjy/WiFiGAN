import scipy.io as scio
import os
from tqdm import tqdm
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
# from datasets.mat_dataset import MatsDataset
from sklearn.model_selection import train_test_split

# Ptot.mat: key = 'Ptot'
# epsono_r.mat: key = 'epsono_r'
# 读取path下mat文件中key对应的数据
def getMat(path, key):
    # path = '/data/shijianyang/data/wifi/1_circular_liquid/1/epsono_r.mat'
    matdata = scio.loadmat(path)
    # print(matdata)
    # print(type(matdata['Ptot']))
    # print(shape(matdata['Ptot']))
    # for i in matadata['epsono_r']:
    #     print(i)
    return matdata[key]

# 将file保存到path
def saveMat(path, file):
    scio.savemat(path, {'data': file})

# 将folder_paths下的epsono和ptot分别处理后存储到epsono_path和ptot_path
def moveMat(folder_paths, epsono_path, ptot_path):
    for folder_path in folder_paths:
        i = 0
        j = 0
        c = folder_path.split('/')[-1][2:]
        print('正在处理类别' + c + '中的文件')
        for foldername, subfolders, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                # print(foldername.split('/')[-1])
                if(filename == 'Ptot.mat'):
                    data = getMat(file_path, 'Ptot')
                    mat_name = c + foldername.split('/')[-1] + '.mat'
                    path = ptot_path + mat_name
                    # print(path)
                    saveMat(path, data)
                    i = i + 1
                if(filename == 'epsono_r.mat'):
                    data = getMat(file_path, 'epsono_r')
                    mat_name = c + foldername.split('/')[-1] + '.mat'
                    path = epsono_path + mat_name
                    # print(path)
                    saveMat(path, data)
                    j = j + 1
        print('已处理' + folder_path.split('/')[-1]+ 'Ptot文件共计' + str(i))
        print('已处理' + folder_path.split('/')[-1]+ 'epsono文件共计' + str(j))

# mat_path为输入的mat，gray_path为输入的要存储的位置
def mat2gray(mat_path, gray_path):
    i = 0
    for foldername, subfolders, filenames in os.walk(mat_path):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            data = getMat(file_path, 'epr_I')
            # data = np.abs(data)
            data = (data - np.min(data))/ (np.max(data) - np.min(data))
            # data[data == data[0][0]] = 255
            data = 255 - data * 255
            data[data < 128] = 0
            data[data >= 128] = 255
            # mask = np.zeros_like(data)
            # print(np.max(data))
            # mask[data > 128] = 1
            # mask[data <= 0.5 * np.max(data)] = 0
            data = data.astype(np.uint8)
            print(data.shape)

            # print(type(data[0][0]))
            # print(data.shape)
            im = Image.fromarray(data)
            im = im.convert('L')
            im = ImageOps.flip(im)
            filename = filename.replace("_epr_l.mat", ".png")
            gray = os.path.join(gray_path, filename)
            # print(gray)
            im.save(gray)
            i = i + 1
            if(i % 1000 == 0):
                print(str(i) + 'files have been processed!')
                # print(data)

# mat_path为输入的mat，rgb_path为输入的要存储的位置
# def mat2rgb(mat_path, rgb_path):
#     i = 0
#     for foldername, subfolders, filenames in os.walk(mat_path):
#         for filename in filenames:
#             file_path = os.path.join(foldername, filename)
#             data = getMat(file_path, 'epr_I')
#             data = np.abs(data)
#             # data[data == data[0][0]] = 255
#             data = data * 255
#             data = data.astype(np.uint8)

#             # print(type(data[0][0]))
#             # print(data.shape)
#             im = Image.fromarray(data)
#             im = im.convert('L')
#             im = ImageOps.flip(im)
#             im = ImageOps.colorize(im, "black", "white")
#             filename = filename.replace("mat", "png")
#             rgb = os.path.join(rgb_path, filename)
#             im.save(rgb)
#             i = i + 1
#             if(i % 1000 == 0):
#                 print(str(i) + 'files have been processed!')

def mat2rgb(mat_path, rgb_path):
    # data = getMat('/data/shijianyang/data/wifi/gan/epsilon/circular_liquid1.mat', 'data')
    i = 0
    for foldername, subfolders, filenames in os.walk(mat_path):
        for filename in tqdm(filenames):
            file_path = os.path.join(foldername, filename)
            data = getMat(file_path, 'data')
            real = np.real(data)
            imag = np.imag(data)
            shape = np.zeros_like(real)
            shape[real == 1] = 1
            shape = shape * 255
            trans = np.zeros_like(imag)
            trans[imag != 0] = 1
            r = real * trans
            rgb_image = np.dstack((r, imag, shape)).astype(np.uint8)

            im = Image.fromarray(rgb_image)
            im = ImageOps.flip(im)
            filename = filename.replace("mat", "png")
            path = os.path.join(rgb_path, filename)
            im.save(path)
            i = i + 1
            # if(i % 1000 == 0):
            #     print(str(i) + 'files have been processed!')

    # print(trans)

def staDis(path):
    i = 0
    all_array = np.zeros((19, 20))
    for foldername, subfolders, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            data = getMat(file_path, 'data')
            if i == 0:
                all_array = data
            # print(data.shape)
            # all_array = np.concatenate((all_array, data), axis=1)
            all_array = np.hstack((all_array, data))
            i = i + 1
            if(i % 1000 == 0):
                print(str(i) + 'files have been processed!')
                print(all_array.shape)
    all_array = all_array.flatten()
    print(all_array.shape)
    hist, edges = np.histogram(all_array, bins=10)
    return hist, edges

def moveFiles(source, target):
    for file in tqdm(source):
        data = getMat(file, 'data')
        filename = file.split('/')[-1]
        file_path = os.path.join(target, filename)
        saveMat(file_path, data)
    print(target + ' has been finished!')

def splitDataset(ptot_path, epsono_path):
    ptot = []
    epsono = []
    for root, dirs, files in os.walk(ptot_path):
        for file in files:
            file_path = os.path.join(root, file)
            ptot.append(file_path)
    for root, dirs, files in os.walk(epsono_path):
        for file in files:
            file_path = os.path.join(root, file)
            epsono.append(file_path)
    ptot_train, ptot_test, epsono_train, epsono_test = train_test_split(ptot, epsono, test_size=0.2, random_state=24)
    # print(len(ptot_train), len(ptot_test), len(epsono_train), len(epsono_test))
    # print(ptot_test[0])
    moveFiles(ptot_train, '/data1/sjy/data/wifigan/train/ptot')
    moveFiles(ptot_test, '/data1/sjy/data/wifigan/test/ptot')
    moveFiles(epsono_train, '/data1/sjy/data/wifigan/train/epsilon')
    moveFiles(epsono_test, '/data1/sjy/data/wifigan/test/epsilon')


# folder_paths = ['/data/shijianyang/data/wifi/h_result', '/data/shijianyang/data/wifi/r_result', '/data/shijianyang/data/wifi/triangle_result']
folder_paths = ['/data/shijianyang/data/wifi/1_circular_liquid', '/data/shijianyang/data/wifi/1_cubic_solid', '/data/shijianyang/data/wifi/1_cylinder', '/data/shijianyang/data/wifi/1_rectangular_liquid', '/data/shijianyang/data/wifi/1_triangular_liquid', '/data/shijianyang/data/wifi/1_triangular_solid']
new_paths = ['/data/shijianyang/data/wifi/2_rectangle_result', '/data/shijianyang/data/wifi/2_triangle_result', '/data/shijianyang/data/wifi/h_result', '/data/shijianyang/data/wifi/circle_result']
epsono_path = '/data/shijianyang/data/wifi/gan/epsilon/'
ptot_path = '/data/shijianyang/data/wifi/gan/ptot/'
mat_path = '/data/shijianyang/data/wifi/gan/epsilon'
m_path = '/data/shijianyang/data/wifi/h_result/1/epsono_r.mat'
gray_path = '/data/shijianyang/data/wifi/epr_mask'
rgb_path = '/data/shijianyang/data/wifi/gan/rgb_epsilon'

train_epsono_path = '/data1/sjy/data/wifigan/train/epsilon/'
train_rgb_path = '/data1/sjy/data/wifigan/train/rgb_epsilon/'
test_epsono_path = '/data1/sjy/data/wifigan/test/epsilon/'
test_rgb_path = '/data1/sjy/data/wifigan/test/rgb_epsilon/'

# 测试文件读取
# a = getMat(m_path, 'epsono_r')
# print(a[0][0].imag)

# 将wifi数据统一移动到/data4psp目录下
# moveMat(folder_paths, epsono_path, ptot_path)

# 将mat转为灰度图存储
# print(os.path.exists(gray_path))
# if not os.path.exists(gray_path):
#     os.makedirs(gray_path)
# mat2gray(mat_path, gray_path)

# 将mat转为rgb图存储
print('train is processing!')
mat2rgb(train_epsono_path, train_rgb_path)
print('test is processing!')
mat2rgb(test_epsono_path, test_rgb_path)

# 统计ptot的数据分布
# hist, edges = staDis(ptot_path)
# plt.title('Ptot_histogram')
# plt.xlabel('Values')
# plt.ylabel('Frequency')
# plt.savefig('ptot_hist.png')
# print(edges)

# data = getMat(m_path, 'epsono_r')
# print(data[data != data[0][0]])

# 查看图片大小
# from PIL import Image

# im = Image.open(rgb_path + '/c1.png')
# print(im.size)

# 测试数据加载
# train_dataset = MatsDataset(source_root='/data/shijianyang/data/wifi/data4psp/ptot', target_root='/data/shijianyang/data/wifi/data4psp/psp/epsono_rgb', opts=None)
# print(len(train_dataset.source_paths))
# print(len(train_dataset.target_paths))

# 划分训练测试数据集
# splitDataset(ptot_path,epsono_path)

# 读取人体和水桶数据
# people = '/data/shijianyang/data/wifi/data4psp/ExperimentData/PeopleStanding/Ptot1.mat'
# mat = getMat(people, 'Ptot')
# print(mat)

# 处理人体和水桶数据
# path = '/data/shijianyang/data/wifi/data4psp/ExperimentData/PeopleStanding'
# save_path = '/data/shijianyang/data/wifi/data4psp/ExperimentData/test'
# Pinc = getMat('/data/shijianyang/data/wifi/data4psp/ExperimentData/PeopleStanding/Pinc.mat', 'Pinc')
# # print(Pinc)
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# # for root, dirs, files in os.walk(path):
# #     for file in files:
# #         if 'Ptot' in file and file != 'Ptot.mat':
# #             print(file)
# #             mat = getMat(os.path.join(root, file), 'Ptot')
# #             new = mat - Pinc
# #             saveMat(os.path.join(save_path, file), new)

# Pinc = getMat('/data/shijianyang/data/wifi/data4psp/ExperimentData/water_and_book_testdata/Pinc.mat', 'Pinc')
# mat = getMat('/data/shijianyang/data/wifi/data4psp/ExperimentData/water_and_book_testdata/Ptot.mat', 'Ptot')
# saveMat(os.path.join(save_path, 'Ptot.mat'), mat - Pinc)