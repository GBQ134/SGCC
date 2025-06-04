#ifndef UTILS2_H
#define UTILS2_H

#include <iostream>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>


#pragma _GLIBCXX_USE_CXX11_ABI=0
using namespace std;

//convenient typedefs
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

void readDataFromFile(std::string filepath, pcl::PointCloud<double> &cloud, pcl::PointCloud<double> &orcloud, std::vector<int> &labels);
void writeOutClusters(std::string filePath, pcl::PointCloud<double> &pointData, pcl::PointCloud<double> &pointOrinData, std::vector<std::vector<int> > &clusters, std::vector<int> &labels);
void stringSplit(const std::string &str, const std::string &splits, std::vector<std::string> &res);
void pointCloudSegmentation(std::string pathS, std::string RGBlabelPath, std::string cvLabelPath);
void planeFit(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float *CV,float *dipAngle, float *dipDirection);

typedef struct txtPoint_3D{
    double x;
    double y;
    double z;
    int R;
    int G;
    int B;
    int i;
    int oi;
}txtPoint_3D;

struct PointXYZRGBLabel:public pcl::PointXYZRGB
{
  int label;                        // 标签
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // 保证内存对齐
};                    // 对齐到16字节

// 告诉PCL如何在新定义的结构体和PointXYZRGB之间进行转换
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZRGBLabel, (float, x, x)(float, y, y)(float, z, z)(int, label, label))

#endif

