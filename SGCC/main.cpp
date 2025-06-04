#include <iostream>
#include <vector>
#include "src/utils2.h"

int main(int argc, char** argv)
{
	std::string inputPath = argv[1];
	std::string RGBlabelPath = argv[2];
	std::string cvLabelPath = argv[3];
	pointCloudSegmentation(inputPath, RGBlabelPath, cvLabelPath);
	
	return (0);
}  