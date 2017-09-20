#ifndef DATA_H
#define DATA_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <numeric>
#include <memory>

#include "matrix_io.h"
#include "utils.h"

#include "model.h"
#include "INoiseModel.h"

// AGE: I dont like the idea of adding this include. this all happens because we have implementation of MatricesData in header.
#include "UnusedNoise.h"

#include "Data.h"
#include "MatrixData.h"

namespace smurff {

struct MatricesData: public MatrixData {
    MatricesData() : total_dim(2) {
        name = "MatricesData";
        noise_ptr = std::unique_ptr<INoiseModel>(new UnusedNoise(this));
    }

    void init_pre() override;
    void init_post() override;
    void setCenterMode(std::string c) override;

    void center(double) override;
    double compute_mode_mean(int,int) override;
    double offset_to_mean(const PVec &pos) const override;

    // add data
    MatrixData &add(const PVec &, std::unique_ptr<MatrixData>);

    // helper functions for noise
    // but
    double sumsq(const SubModel &) const override { assert(false); return NAN; }
    double var_total() const override { return NAN; }
    double train_rmse(const SubModel &) const override;

    // update noise and precision/mean
    void update(const SubModel &model) override;
    void get_pnm(const SubModel &,int,int,VectorNd &, MatrixNNd &) override;
    void update_pnm(const SubModel &model, int mode) override;

    //-- print info
    std::ostream &info(std::ostream &os, std::string indent) override;
    std::ostream &status(std::ostream &os, std::string indent) const override;

    // accumulate on data in a block
    template<typename T, typename F>
    T accumulate(T init, F func) const {
        return std::accumulate(blocks.begin(), blocks.end(), init,
            [func](T s, const Block &b) -> T { return  s + (b.data().*func)(); });
    }

    int    nnz() const override { return accumulate(0, &MatrixData::nnz); }
    int    nna() const override { return accumulate(0, &MatrixData::nna); }
    double sum() const override { return accumulate(.0, &MatrixData::sum); }
    PVec   dim() const override { return total_dim; }

  private:
    struct Block {
        friend struct MatricesData;
        // c'tor
        Block(PVec p, std::unique_ptr<MatrixData> c)
            : _pos(p), _start(2),  m(std::move(c)) {}

        // handy position functions
        const PVec start() const  { return _start; }
        const PVec end() const  { return start() + dim(); }
        const PVec dim() const { return data().dim(); }
        const PVec pos()  const { return _pos; }

        int start(int mode) const { return start().at(mode); }
        int end(int mode) const { return end().at(mode); }
        int dim(int mode) const { return dim().at(mode); }
        int pos(int mode) const { return pos().at(mode); }

        MatrixData &data() const { return *m; }

        bool in(const PVec &p) const { return p.in(start(), end()); }
        bool in(int mode, int p) const { return p >= start(mode) && p < end(mode); }

        SubModel submodel(const SubModel &model) const;

      private:
        PVec _pos, _start;
        std::unique_ptr<MatrixData> m;

    };
    std::vector<Block> blocks;

    template<typename Func>
    void apply(int mode, int p, Func f) const {
        for(auto &b : blocks) if (b.in(mode, p)) f(b);
    }

    const Block& find(const PVec &p) const {
        return *std::find_if(blocks.begin(), blocks.end(), [p](const Block &b) -> bool { return b.in(p); });
    }

    int nview(int mode) const override {
        return mode_dim.at(mode).size();
    }
    int view(int mode, int pos) const override {
        assert(pos < MatrixData::dim(mode));
        const auto &v = mode_dim.at(mode);
        int off = 0;
        for(int i=0; i<nview(mode); ++i) {
           off += v.at(i);
           if (pos < off) return i;
        }
        assert(false);
        return -1;
    }

    std::vector<std::vector<int>> mode_dim;
    PVec total_dim;
 };


template<typename YType>
struct MatrixDataTempl : public MatrixData {
    MatrixDataTempl(YType Y) : Y(Y) {}

    //init and center
    void init_pre() override;

    PVec   dim() const override { return PVec(Y.cols(), Y.rows()); }
    int    nnz() const override { return Y.nonZeros(); }
    double sum() const override { return Y.sum(); }

    double offset_to_mean(const PVec &pos) const override;

    double var_total() const override;
    double sumsq(const SubModel &) const override;

    YType Y;
    std::vector<YType> Yc; // centered versions

};

struct ScarceMatrixData : public MatrixDataTempl<SparseMatrixD> {
    //-- c'tor
    ScarceMatrixData(SparseMatrixD Y)
        : MatrixDataTempl<SparseMatrixD>(Y)
    {
        name = "ScarceMatrixData [with NAs]";
    }

    void init_pre() override;
    void center(double) override;
    double compute_mode_mean(int,int) override;

    double train_rmse(const SubModel &) const override;

    std::ostream &info(std::ostream &os, std::string indent) override;

    void get_pnm(const SubModel &,int,int,VectorNd &, MatrixNNd &) override;
    void update_pnm(const SubModel &,int) override {}

    int    nna()             const override { return size() - nnz(); }

  private:
    int num_empty[2] = {0,0};

};

struct ScarceBinaryMatrixData : public ScarceMatrixData {
    //-- c'tor
    ScarceBinaryMatrixData(SparseMatrixD &Y) : ScarceMatrixData(Y)
    {
        name = "ScarceBinaryMatrixData [containing 0,1,NA]";
    }

    void get_pnm(const SubModel &,int,int,VectorNd &, MatrixNNd &) override;
    void update_pnm(const SubModel &,int) override {};

    int    nna()   const override { return size() - nnz(); }

};

template<class YType>
struct FullMatrixData : public MatrixDataTempl<YType> {
    //-- c'tor
    FullMatrixData(YType Y) : MatrixDataTempl<YType>(Y)
    {
        this->name = "MatrixData [fully known]";
    }


    void get_pnm(const SubModel &,int,int,VectorNd &, MatrixNNd &) override;
    void update_pnm(const SubModel &,int) override;

    int    nna()   const override { return 0; }

  private:
    Eigen::MatrixXd VV[2];

    double compute_mode_mean(int,int) override;
};

struct DenseMatrixData : public FullMatrixData<Eigen::MatrixXd> {
    //-- c'tor
    DenseMatrixData(Eigen::MatrixXd Y) : FullMatrixData<Eigen::MatrixXd>(Y)
    {
        this->name = "DenseMatrixData [fully known]";
    }
    void center(double) override;
    double train_rmse(const SubModel &) const override;
};

struct SparseMatrixData : public FullMatrixData<Eigen::SparseMatrix<double>> {
    //-- c'tor
    SparseMatrixData(Eigen::SparseMatrix<double> Y) : FullMatrixData<Eigen::SparseMatrix<double>>(Y)
    {
        this->name = "SparseMatrixData [fully known]";
    }
    void center(double) override;
    double train_rmse(const SubModel &) const override;
};

}; // end namespace smurff

#endif
