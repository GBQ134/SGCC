import open3d as o3d
import numpy as np
import pandas as pd

if __name__ == "__main__":
    points = pd.read_csv("C:\\Users\\GBQ\\Desktop\\juleibankuai\\5000wCV.TXT", sep=',', header=None)

    ptnp = np.array(points)
    seglabels = ptnp[:, 6] #假设第7列是分割算法得到的标签列
    lbtypes = np.unique(seglabels)  #获得有多少个分割标签唯一值
    centerinteg = []    #储存每一个分割标签的坐标、特征计算结果
    for i in range(len(lbtypes)):   #根据分割标签数量循环计算
        idxs = np.where(seglabels==lbtypes[i]) #返回第i个分割标签的点索引

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ptnp[idxs][:,:3])
        center = list(pcd.get_center())  #计算分割编号lbtypes[i]点云的质心坐标

        avef1 = np.mean(ptnp[idxs][:,3])
        avef2 = np.mean(ptnp[idxs][:,4])
        avef3 = np.mean(ptnp[idxs][:,5])    #计算分割编号lbtypes[i]点云的若干特征均值

        labelcounts = np.bincount(ptnp[idxs][:,-1].astype('uint8'))
        most_common_index = np.argmax(labelcounts).astype('uint8')  #计算分割编号lbtypes[i]点云的真值标签，以出现最多的标签为返回值

        center.append(avef1)
        center.append(avef2)
        center.append(avef3)
        center.append(most_common_index)    #将计算的特征和返回的真值标签附于质心坐标后，作为分割斑块的（x y z 特征 标签） 行
        # center.append(lbtypes[i])  # 添加斑块号到最后一列

        centerinteg.append(center)  #存储该标签对应的行

    centerintegnp = np.array(centerinteg)
    pdout = pd.DataFrame(centerintegnp)

    pdout.to_csv("5000.txt", sep=',', index=0,header=None)   #列表转换数组并保存


