#pragma once

void writeToCSVfile(const std::string& filename, const Eigen::MatrixXd& matrix);
void writeToCSVstream(std::ostream& out, const Eigen::MatrixXd& matrix);

void readFromCSVfile(const std::string& filename, Eigen::MatrixXd& matrix);
void readFromCSVstream(std::istream& in, Eigen::MatrixXd& matrix);

void write_ddm(const std::string& filename, const Eigen::MatrixXd& matrix);
void write_ddm(std::ostream& out, const Eigen::MatrixXd& matrix);

void read_ddm(const std::string& filename, Eigen::MatrixXd& matrix);
void read_ddm(std::istream& in, Eigen::MatrixXd& matrix);

void read_dense(const std::string& fname, Eigen::MatrixXd& X);
void read_dense(std::istream& in, DenseMatrixType denseMatrixType, Eigen::MatrixXd& X);

void read_dense(const std::string& fname, Eigen::VectorXd &);
void read_dense(std::istream& in, DenseMatrixType denseMatrixType, Eigen::VectorXd& V);

void read_sparse(const std::string& fname, Eigen::SparseMatrix<double> &);

void write_dense(const std::string& fname, const Eigen::MatrixXd&);
void write_dense(std::ostream& out, DenseMatrixType denseMatrixType, const Eigen::MatrixXd&);