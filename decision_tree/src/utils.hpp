#include <iostream>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <map>




class InfoGain
{
    public:

        double information_gain(const Eigen::MatrixXd& x, const Eigen::VectorXi& y, const std::vector<int>& node_inxs, int feat_inx, double threshold)
        {
            std::vector<int> left_inxs;
            std::vector<int> right_inxs;

            split_data(x, left_inxs, right_inxs, node_inxs, feat_inx, threshold);

            
            Eigen::VectorXi y_node(node_inxs.size());
            Eigen::VectorXi y_left(left_inxs.size());
            Eigen::VectorXi y_right(right_inxs.size());

            for (int i = 0; i < node_inxs.size(); i++)
            {
                y_node(i) = y(node_inxs[i]);    
            }

            for (int i = 0; i < left_inxs.size(); i++)
            {
                y_left(i) = y(left_inxs[i]);    
            }

            for (int i = 0; i < right_inxs.size(); i++)
            {
                y_right(i) = y(right_inxs[i]);    
            }
          
            double info_gain = 0.0;
            double node_entropy = compute_entropy(y_node);
            double left_entropy = compute_entropy(y_left);
            double right_entropy = compute_entropy(y_right);

            double w_left = static_cast<double>(y_left.size()) / y_node.size();
            double w_right = static_cast<double>(y_right.size()) / y_node.size();


            double w_entropy = (w_left * left_entropy) + (w_right * right_entropy);

            return node_entropy - w_entropy;

        }
        
       
        void split_data(const Eigen::MatrixXd& x, std::vector<int>& left_inxs, std::vector<int>& right_inxs, const std::vector<int>& node_inxs, int feat_inx, double threshold)
        {
            for (int inx: node_inxs)
            {
                if (x(inx, feat_inx) <= threshold)
                {
                    left_inxs.push_back(inx);
                }
                else
                {
                    right_inxs.push_back(inx);
                }
            }

        }

        int best_split(const Eigen::MatrixXd& x, const Eigen::VectorXi& y, const std::vector<int>& node_inxs, const double threshold)
        {
           
            int num_feat = x.cols();
            int best_feat_inx = -1;
            double max_info_gain = -1.0;

            for (int f = 0; f < num_feat; f++)
            {
                double info_gain = information_gain(x, y, node_inxs, f, threshold);

                if (info_gain > max_info_gain)
                {
                    max_info_gain = info_gain;
                    best_feat_inx = f;
                }

            }

            return best_feat_inx;
        }

    private:

        double compute_entropy(const Eigen::VectorXi& y)
        {

            if (y.size() == 0)
            {
                return 0.0;
            }
            double num_true = 0.0;
            for (int i = 0; i < y.size(); ++i)
            {
                if (y(i) == 1) 
                {
                    num_true++;
                }
            }
            double p = num_true / static_cast<double>(y.size());

            if (p == 0.0 || p == 1.0)
            {
                return 0.0;
            }
            else
            {
                return - p * std::log2(p) - (1.0 - p) * std::log2(1.0 - p);
            
            }

        }

};



struct TreeNode
{
    std::vector<int> left_inxs;
    std::vector<int>right_inxs;
    int best_feat;

    TreeNode(const std::vector<int>& left, const std::vector<int>& right, int feat)
        : left_inxs(left), right_inxs(right), best_feat(feat) {}
};



class DecisionTree
{
    public:

        DecisionTree(int max_depth = 2, const double threshold = 0.5)
            : max_depth(max_depth), threshold(threshold) {} 
        
      
        void run(const Eigen::MatrixXd& x, const Eigen::VectorXi& y, const std::vector<int>& node_inxs, const std::string& branch_name, int max_depth, int curr_depth, const double threshold)
        {
            if (curr_depth == max_depth)
            {
                std::string form(curr_depth, ' ');
                form += std::string(curr_depth, '-');

                std::cout << form << " " << branch_name << " leaf node with indices: ";
                
                for (int idx : node_inxs)
                    std::cout << idx << " ";
                
                std::cout << std::endl;
                return;
            }

            int best_feat_inx = gain_calculator.best_split(x, y, node_inxs, threshold);

            if (best_feat_inx == -1)
            {
                std::string form(curr_depth, ' ');
                form += std::string(curr_depth, '-');

                std::cout << form << " " << branch_name << " leaf node with indices: ";

                for (int idx : node_inxs)
                    std::cout << idx << " ";
                std::cout << std::endl;
                return;
            }

            std::string form(curr_depth, '-');
            std::cout << form << " Depth " << curr_depth << ", " << branch_name << ": Split on feature: " << best_feat_inx << std::endl;

            std::vector<int> left_inxs;
            std::vector<int> right_inxs;

            gain_calculator.split_data(x, left_inxs, right_inxs, node_inxs, best_feat_inx, threshold);
            tree_nodes.emplace_back(left_inxs, right_inxs, best_feat_inx); // store the branches
            
            run(x, y, left_inxs, "left", max_depth, curr_depth+1, threshold);
            run(x, y, right_inxs, "right", max_depth, curr_depth+1, threshold);

        }

        // TODO: add a predict function!!!


    private:
        int max_depth;
        double threshold;
        InfoGain gain_calculator;
        std::vector<TreeNode> tree_nodes;


};