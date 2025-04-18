#include <Eigen/Dense>
#include <iostream>

int main() {
    Eigen::VectorXd x(3);
    Eigen::VectorXd y(3);

    x << 1, 2, 3;
    y << 2, 4, 6;

    double slope = (x.dot(y)) / (x.dot(x));
    std::cout << "Estimated slope: " << slope << std::endl;

    return 0;
}
