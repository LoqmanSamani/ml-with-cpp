#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <random>
#include "utils.hpp"



int main()
{ 
    Eigen::VectorXd n(4);
    Eigen::VectorXd m(4);
    n << 1, 2, 3, 4;
    m << 5, 6, 7, 8;
    std::cout << m.transpose() * n << std::endl;

    return 1;
}
