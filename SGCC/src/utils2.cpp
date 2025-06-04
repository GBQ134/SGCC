#include <iostream>
#include <string>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include "utils2.h"
#include "utils.h"
#include "PointGrowAngleOnly.h"
#include "PointGrowAngleDis.h"
#include "ClusterGrowPLinkage.h"
#pragma _GLIBCXX_USE_CXX11_ABI=0
using namespace std;

void readDataFromFile(std::string filepath, PointCloud<double> &cloud, PointCloud<double> &orcloud, std::vector<int> &labels)
{
    cloud.pts.reserve(10000000);
    cout << "Reading data ..." << endl;

    // 1. read in point data
    std::ifstream ptReader(filepath);
    std::vector<cv::Point3d> lidarPoints;
    double x = 0, y = 0, z = 0;
    double nx, ny, nz;
    double na, nb, nc;
    char t;
    string strs;


    int a = 0, b = 0, c = 0;
    int labelIdx = 0;
//    int count = 0;
//    int countTotal = 0;
    if (ptReader.is_open())
    {
        while (!ptReader.eof())
        {
            //ptReader >> x >> y >> z >> a >> b >> c >> labelIdx;
            //ptReader >> x >> y >> z >> a >> b >> c >> color;
            //ptReader >> x >> y >> z >> color >> a >> b >> c;
//            ptReader >> x >> y >> z >> a >> b >> c ;
//            ptReader >> x >> t >> y >>t >> z >> t >> a >> t >>b >> t >> c;
            ptReader >> x >> t >> y >>t >> z>>t>>nx>>t>>ny>>t>>nz>>t>>na>>t>>nb>>t>>nc>>t>>labelIdx>>strs;
            //ptReader >> x >> y >> z >> nx >> ny >> nz;

            cloud.pts.push_back(PointCloud<double>::PtData(x, y, z));
            orcloud.pts.push_back(PointCloud<double>::PtData(x, y, z));
            labels.push_back(labelIdx);

        }
        ptReader.close();
    }
    std::cout<< "Total num of points: \n" + to_string(cloud.pts.size())<<std::endl ;
}


void writeOutClusters(std::string filePath, PointCloud<double> &pointData, PointCloud<double> &pointOrinData, std::vector<std::vector<int> > &clusters, std::vector<int> &labels)
{
    std::vector<cv::Scalar> colors(100);

    for (int i = 0; i<100; ++i)
    {
        int R = rand() % 255;
        int G = rand() % 255;
        int B = rand() % 255;
        colors[i] = cv::Scalar(B, G, R);
    }

    FILE *fp = fopen(filePath.c_str(), "w");
    for (int i = 0; i<clusters.size(); ++i)
    {
        int R = rand() % 255;
        int G = rand() % 255;
        int B = rand() % 255;
        for (int j = 0; j<clusters[i].size(); ++j)
        {
            int idx = clusters[i][j];
            for (int k=0; k<pointOrinData.pts.size(); ++k)
            {
            if (pointData.pts[idx].x==pointOrinData.pts[k].x && pointData.pts[idx].y==pointOrinData.pts[k].y && pointData.pts[idx].z==pointOrinData.pts[k].z)
            {
            fprintf(fp, "%f,%f,%f,%d,%d,%d,%d,%d\n",
                pointData.pts[idx].x, pointData.pts[idx].y, pointData.pts[idx].z, R, G, B,i,labels[k]);
            }
            }


        }
    }

    fclose(fp);
}

void stringSplit(const std::string &str, const std::string &splits, std::vector<std::string> &res)
{
    if(str=="")
    {
        return;
    }
    std::string strs = str+splits;
    size_t pos = strs.find(splits);
    int step = splits.size();
    while (pos!=strs.npos)
    {
        std::string temp = strs.substr(0, pos);
        res.push_back(temp);
        strs = strs.substr(pos+step, strs.size());
        pos = strs.find(splits);
    }
}

void pointCloudSegmentation(std::string pathS, std::string RGBlabelPath, std::string cvLabelPath)
{   
    std::vector<std::string> resultVec;
    stringSplit(pathS, "/", resultVec);

   
    PointCloud<double> pointData;
    PointCloud<double> pointOrinData;
    std::vector<int> lbs;
    readDataFromFile(pathS, pointData, pointOrinData, lbs);
    
    // step2: build kd-tree
    std::cout<<"building kd-tree pca normal calculation create linkage"<<std::endl;
    int k = 100;
    std::vector<PCAInfo> pcaInfos;
    PCAFunctions pcaer;
    pcaer.PCA(pointData, 100, pcaInfos);

    // step3: run point segmentation algorithm
    int algorithm = 0;
    double lambdaSTD=0.0;
    int clusterNumber = 0;
    int finalPlaneNum = 0;
    std::vector<std::vector<int>> clusters;

    // Algorithm1: segmentation via PLinkage based clustering
    if (algorithm == 0)
    {

        double theta = 90.0 / 180.0 * CV_PI;
        PLANE_MODE planeMode = SURFACE;               // PLANE  SURFACE
        ClusterGrowPLinkage segmenter(k, theta, planeMode);
        segmenter.setData(pointData, pcaInfos);
        segmenter.run(clusters, lambdaSTD, clusterNumber, finalPlaneNum);
        // std::cout<<"lambdaSTD:"+to_string(lambdaSTD)<<std::endl;
        // std::cout<<"Clustering"<<std::endl;
        // std::cout<<"Clusters number: " + to_string(clusterNumber)<<std::endl;
        // std::cout<<"Patch merging"<<std::endl;
        // std::cout<<"final plane's number: "+to_string(finalPlaneNum)<<std::endl;
    }
    // Algorithm2: segmentation via normal angle similarity
    else if (algorithm == 1)
    {
        double theta = 5.0 / 180.0 * CV_PI;
        double percent = 0.75;
        PointGrowAngleOnly segmenter(theta, percent);
        segmenter.setData(pointData, pcaInfos);
        segmenter.run(clusters);
    }
    // Algorithm3: segmentation via normal angle similarity and point-plane distance
    else
    {
        double theta = 10.0 / 180.0 * CV_PI;
        int RMin = 10;  // minimal number of points per cluster
        PointGrowAngleDis segmenter(theta, RMin);
        segmenter.setData(pointData, pcaInfos);
        segmenter.run(clusters);
    }

    // step4: write out result
    writeOutClusters(RGBlabelPath, pointData, pointOrinData, clusters, lbs);

    std::string readPath = RGBlabelPath;
    FILE *fp_txt;
    txtPoint_3D txtPoint;
    vector<txtPoint_3D> txtPoints;
    fp_txt = fopen(readPath.c_str(),"r");

    if (fp_txt == NULL){
        cerr<<"open error!"<<endl;
    }

    while (fscanf(fp_txt, "%lf,%lf,%lf,%d,%d,%d,%d,%d", &txtPoint.x, &txtPoint.y, &txtPoint.z, &txtPoint.R, &txtPoint.G, &txtPoint.B, &txtPoint.i, &txtPoint.oi) != EOF){
        //cout<<txtPoint.x<<" "<<txtPoint.y<<" "<<txtPoint.z<<endl;
        txtPoints.push_back(txtPoint);
    }


    pcl::PointCloud<PointXYZRGBLabel>::Ptr clouds(new pcl::PointCloud<PointXYZRGBLabel>);
    PointXYZRGBLabel cloud;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr showclouds(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointXYZRGB showcloud;

    std::vector<int>orinLabels;

    for (size_t i = 0; i < txtPoints.size(); ++i){
        showcloud.x = txtPoints[i].x;
        showcloud.y = txtPoints[i].y;
        showcloud.z = txtPoints[i].z;
        uint32_t rgb = (static_cast<uint32_t>(txtPoints[i].R) << 16 |
                static_cast<uint32_t>(txtPoints[i].G) << 8 | static_cast<uint32_t>(txtPoints[i].B));
        showcloud.rgb = *reinterpret_cast<float*>(&rgb);
        showclouds->points.push_back(showcloud);

        cloud.x = txtPoints[i].x;
        cloud.y = txtPoints[i].y;
        cloud.z = txtPoints[i].z;
        cloud.rgb = *reinterpret_cast<float*>(&rgb);
        cloud.label = txtPoints[i].i;
        orinLabels.push_back(txtPoints[i].oi);
        clouds->points.push_back(cloud);

    }

    pcl::PointXYZ planePoint;
    pcl::PointCloud<pcl::PointXYZ>::Ptr planePoints(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<int> labels;

    float cv=0.0;
    float dipAngle = 0.0;
    float dipDirection = 0.0;
    std::vector<float> cvs;
    std::vector<float> dipAngles;
    std::vector<float> dipDirections;


    string filePathcv = cvLabelPath;
    FILE *fp = fopen(filePathcv.c_str(), "w");


    for(int i=0; i<finalPlaneNum; ++i)
    {
        for (size_t j=0; j<clouds->points.size(); ++j)
        {

            if(clouds->points[j].label==i)
            {

                planePoint.x = clouds->points[j].x;
                planePoint.y = clouds->points[j].y;
                planePoint.z = clouds->points[j].z;
                planePoints->points.push_back(planePoint);
                labels.push_back(clouds->points[j].label);



            }

        }

        planeFit(planePoints, &cv, &dipAngle,&dipDirection);
        cvs.push_back(cv);
        dipAngles.push_back(dipAngle);
        dipDirections.push_back(dipDirection);
    }

    for(int i=0; i<finalPlaneNum; ++i)
    {
        for (size_t j=0; j<clouds->points.size(); ++j)
        {

            if(clouds->points[j].label==i)
            {

                fprintf(fp, "%f,%f,%f,%f,%f,%f,%d,%d\n",
                        planePoints->points[j].x, planePoints->points[j].y, planePoints->points[j].z, cvs[i], dipAngles[i],dipDirections[i],labels[j], orinLabels[j]);

            }

        }


    }

    fclose(fp);


    std::cout << "Saved " << showclouds->points.size () << " data points" << std::endl;

}

void planeFit(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float *CV,float *dipAngle, float *dipDirection)
{
   pcl::SACSegmentation<pcl::PointXYZ> seg;
   pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
   pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

   // 设置SACSegmentation参数
   seg.setOptimizeCoefficients(true);
   seg.setModelType(pcl::SACMODEL_PLANE);
   seg.setMethodType(pcl::SAC_RANSAC);
//   seg.setMaxIterations(1000);
   seg.setDistanceThreshold(0.01);

   // 执行平面拟合
   seg.setInputCloud(cloud);
   seg.segment(*inliers, *coefficients);

   if (inliers->indices.size() == 0)
       {
           // 如果没有找到平面，返回错误或默认值
           *dipAngle = -1.0f;
           *dipDirection = -1.0f;

       }

   float a = coefficients->values[0];
   float b = coefficients->values[1];
   float c = coefficients->values[2];
   float d = coefficients->values[3];

   // 计算法线向量的方向余弦（归一化）
       float nx = a;
       float ny = b;
       float nz = c;
       float norm = sqrt(nx * nx + ny * ny + nz * nz);
       nx /= norm;
       ny /= norm;
       nz /= norm;

       // 倾角是法线向量与z轴（通常是垂直方向）的夹角
       *dipAngle = acos(nz) * 180.0f / M_PI; // 转换为度

       // 倾向是法线向量在xy平面上的投影与x轴的夹角
       // 注意：这里我们假设正北方向是x轴的正方向，实际情况可能需要调整
       if (ny == 0 && nx > 0)
           *dipDirection = 0.0f; // 避免除以零
       else if (ny == 0 && nx < 0)
           *dipDirection = 180.0f;
       else
           *dipDirection = atan2(ny, nx) * 180.0f / M_PI; // 转换为度，并考虑象限

       if (*dipDirection < 0)
           *dipDirection += 360.0f; // 保证倾向在0到360度之间


   float distance_sum = 0.0;
   int num_points = 0;
   std::vector<float> distances;
   distances.resize(cloud->points.size()); // 预先分配空间

   // 使用OpenMP并行化循环
   #pragma omp parallel for reduction(+:distance_sum) schedule(dynamic)
   for (size_t i = 0; i < cloud->points.size(); ++i)
   {
       float x = cloud->points[i].x;
       float y = cloud->points[i].y;
       float z = cloud->points[i].z;
       float distance = fabs(a * x + b * y + c * z + d) / sqrt(a * a + b * b + c * c);
       distances[i] = distance; // 存储距离到vector中
       distance_sum += distance;
       #pragma omp atomic
       num_points++; // 使用原子操作来安全地增加计数器
   }

   float average_distance = distance_sum / num_points;

   float SDsum = 0.0;
   for (const auto& value : distances)
   {
       SDsum += pow(value - average_distance, 2);
   }

   float SD = sqrt(SDsum / (num_points - 1)); // 如果cloud是一个样本集

   *CV = SD / average_distance;

}