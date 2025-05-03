#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <cmath>
#include <map>
#include <algorithm> 




class InfoGain 
{
    public:

        double information_gain(const Eigen::MatrixXd& x, const Eigen::VectorXi& y, const std::vector<int>& node_inxs, int feat_inx, double threshold)
        {
            std::vector<int> left_inxs;
            std::vector<int> right_inxs;

            split_data(x, left_inxs, right_inxs, node_inxs, feat_inx, threshold);

            // check for empty splits
            if (left_inxs.empty() || right_inxs.empty()) 
            {
                return 0.0;
            }

            // validate labels. ensure y contains non-negative integers
            for (int idx : node_inxs) 
            {
                if (y(idx) < 0) 
                {
                    throw std::runtime_error("Invalid label: " + std::to_string(y(idx)));
                }
            }

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
            for (int inx : node_inxs) 
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

        struct SplitResult 
        {
            int best_feat_inx;
            double best_threshold;
            double max_info_gain;

        };

        SplitResult best_split(const Eigen::MatrixXd& x, const Eigen::VectorXi& y, const std::vector<int>& node_inxs, int min_samples_split = 2) 
        {
            SplitResult result = {-1, 0.0, -1.0};
            int num_feat = x.cols();

            if (node_inxs.size() < min_samples_split) 
            {
                return result; // too few samples to split
            }

            for (int f = 0; f < num_feat; f++) 
            {
            
                std::vector<double> values;
                for (int idx : node_inxs) 
                {
                    values.push_back(x(idx, f));
                }
                std::sort(values.begin(), values.end());
                
                std::vector<double> thresholds;
                for (size_t i = 1; i < values.size(); i++) 
                {
                    if (values[i] != values[i-1]) 
                    {
                        thresholds.push_back((values[i] + values[i-1]) / 2.0);
                    }
                }
                if (thresholds.empty()) 
                {
                    continue;
                }

                for (double thresh : thresholds) 
                {
                    double info_gain = information_gain(x, y, node_inxs, f, thresh);

                    if (info_gain > result.max_info_gain) 
                    {
                        result.max_info_gain = info_gain;
                        result.best_feat_inx = f;
                        result.best_threshold = thresh;
                    }
                }
            }

            return result;
        }

    private:

        double compute_entropy(const Eigen::VectorXi& y) 
        {
            if (y.size() == 0)
            {
                return 0.0;
            }

            std::map<int, int> class_counts;
            for (int i = 0; i < y.size(); ++i) 
            {
                class_counts[y(i)]++;
            }

            double entropy = 0.0;
            for (const auto& pair : class_counts) 
            {
                double p = static_cast<double>(pair.second) / y.size();
                if (p > 0.0) 
                { 
                    entropy -= p * std::log2(p);
                }
            }
            return entropy;
        }
};


struct TreeNode 
{
    std::vector<int> left_inxs;
    std::vector<int> right_inxs;
    int best_feat;
    double threshold; 
    int majority_label;
    bool is_leaf;
    int left_child_idx;
    int right_child_idx;

    TreeNode(const std::vector<int>& left, const std::vector<int>& right, int feat, double thresh, int label, bool leaf = false)
        : left_inxs(left), right_inxs(right), best_feat(feat), threshold(thresh), majority_label(label), is_leaf(leaf),
          left_child_idx(-1), right_child_idx(-1) {}
};


class DecisionTree 
{
    public:

        DecisionTree(int max_depth = 2, double threshold = 0.5, int min_samples_split = 2)
            : max_depth(max_depth), default_threshold(threshold), min_samples_split(min_samples_split) {}
    
        void run(const Eigen::MatrixXd& x, const Eigen::VectorXi& y, const std::vector<int>& node_inxs, const std::string& branch_name, int curr_depth) 
        {
            if (node_inxs.size() < min_samples_split) 
            {
                int majority = compute_majority_label(y, node_inxs);
                std::string form(curr_depth, ' ');
                form += std::string(curr_depth, '-');
                std::cout << form << " " << branch_name << " leaf node (too few samples) with indices: ";
                for (int idx : node_inxs) std::cout << idx << " ";
                std::cout << " | majority label: " << majority << std::endl;
                tree_nodes.emplace_back(std::vector<int>(), std::vector<int>(), -1, 0.0, majority, true);
                return;
            }
    
            bool is_pure = true;
            int first_label = y(node_inxs[0]);
            for (int idx : node_inxs) {
                if (y(idx) != first_label) 
                {
                    is_pure = false;
                    break;
                }
            }
            if (is_pure) 
            {
                std::string form(curr_depth, ' ');
                form += std::string(curr_depth, '-');
                std::cout << form << " " << branch_name << " leaf node (pure) with indices: ";
                for (int idx : node_inxs) std::cout << idx << " ";
                std::cout << " | majority label: " << first_label << std::endl;
                tree_nodes.emplace_back(std::vector<int>(), std::vector<int>(), -1, 0.0, first_label, true);
                return;
            }
    
            if (curr_depth == max_depth) 
            {
                int majority = compute_majority_label(y, node_inxs);
                std::string form(curr_depth, ' ');
                form += std::string(curr_depth, '-');
                std::cout << form << " " << branch_name << " leaf node (max depth) with indices: ";
                for (int idx : node_inxs) std::cout << idx << " ";
                std::cout << " | majority label: " << majority << std::endl;
                tree_nodes.emplace_back(std::vector<int>(), std::vector<int>(), -1, 0.0, majority, true);
                return;
            }
    
            InfoGain::SplitResult split = gain_calculator.best_split(x, y, node_inxs, min_samples_split);
    
            if (split.best_feat_inx == -1) 
            {
                int majority = compute_majority_label(y, node_inxs);
                std::string form(curr_depth, ' ');
                form += std::string(curr_depth, '-');
                std::cout << form << " " << branch_name << " leaf node (no split) with indices: ";
                for (int idx : node_inxs) std::cout << idx << " ";
                std::cout << " | majority label: " << majority << std::endl;
                tree_nodes.emplace_back(std::vector<int>(), std::vector<int>(), -1, 0.0, majority, true);
                return;
            }
    
            std::string form(curr_depth, '-');
            std::cout << form << " Depth " << curr_depth << ", " << branch_name << ": Split on feature " << split.best_feat_inx
                      << " with threshold " << split.best_threshold << std::endl;
    
            std::vector<int> left_inxs;
            std::vector<int> right_inxs;
            gain_calculator.split_data(x, left_inxs, right_inxs, node_inxs, split.best_feat_inx, split.best_threshold);
    
            
            size_t current_node_idx = tree_nodes.size();
            tree_nodes.emplace_back(left_inxs, right_inxs, split.best_feat_inx, split.best_threshold, -1, false);
    
            
            run(x, y, left_inxs, "left", curr_depth + 1);
            run(x, y, right_inxs, "right", curr_depth + 1);
    
            
            if (current_node_idx + 1 < tree_nodes.size()) 
            {
                tree_nodes[current_node_idx].left_child_idx = current_node_idx + 1;
            }
            
            size_t left_subtree_size = 0;
            for (size_t i = current_node_idx + 1; i < tree_nodes.size() && !tree_nodes[i].is_leaf && (tree_nodes[i].left_inxs != right_inxs); i++) 
            {
                left_subtree_size++;
            }
            if (current_node_idx + 1 + left_subtree_size < tree_nodes.size()) 
            {
                tree_nodes[current_node_idx].right_child_idx = current_node_idx + 1 + left_subtree_size;
            }
        }
    
        Eigen::VectorXi predict(const Eigen::MatrixXd& x, const Eigen::VectorXi& y) 
        {
            if (tree_nodes.empty()) 
            {
                throw std::runtime_error("Tree not trained yet!");
            }
    
            Eigen::VectorXi predictions(x.rows());
            for (Eigen::Index i = 0; i < x.rows(); ++i) 
            {
                predictions(i) = predict_single(x.row(i), y, 0); 
            }
            return predictions;
        }
    
    private:

        int max_depth;
        double default_threshold;
        int min_samples_split;
        InfoGain gain_calculator;
        std::vector<TreeNode> tree_nodes;
    
        int compute_majority_label(const Eigen::VectorXi& y, const std::vector<int>& indices) 
        {
            if (indices.empty()) 
            {
                return -1; 
            }
            std::map<int, int> class_counts;
            for (int idx : indices) 
            {
                class_counts[y(idx)]++;
            }
            int max_count = 0;
            int majority_label = -1;
            for (const auto& pair : class_counts) 
            {
                if (pair.second > max_count) 
                {
                    max_count = pair.second;
                    majority_label = pair.first;
                }
            }
            return majority_label;
        }
    
        int predict_single(const Eigen::MatrixXd::ConstRowXpr& x, const Eigen::VectorXi& y, size_t node_idx) 
        {
            const TreeNode& node = tree_nodes[node_idx];
            if (node.is_leaf) 
            {
                return node.majority_label;
            }
    
            if (x(node.best_feat) <= node.threshold) 
            {
                if (node.left_child_idx == -1) 
                {
                    return compute_majority_label(y, node.left_inxs);
                }
                return predict_single(x, y, node.left_child_idx);
            } 
            else 
            {
                if (node.right_child_idx == -1) 
                {
                    return compute_majority_label(y, node.right_inxs);
                }
                return predict_single(x, y, node.right_child_idx);
            }
        }
    };


    
#endif