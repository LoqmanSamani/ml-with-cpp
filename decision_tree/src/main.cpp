#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <string>
#include "data_proc.hpp"
#include "utils.hpp"







int main()
{
    Eigen::MatrixXd m(3,3);
    m << 1,2,3,4,5,6,7,8,9;
    std::cout <<m << std::endl;
    

    return 0;
}
