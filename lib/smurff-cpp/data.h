#ifndef DATA_H
#define DATA_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <numeric>
#include <memory>

#include "noisemodels.h"
#include "matrix_io.h"
#include "utils.h"

namespace smurff {

struct Data {
    Data() : center_mode(CENTER_INVALID) {}

    // init
    virtual void init_base() = 0;
    virtual void center(double upper_mean) = 0;
    virtual void init();

    // helper functions for noise
    virtual double sumsq(const SubModel &) const = 0;
    virtual double var_total() const = 0;

    // update noise and precision/mean
    virtual double train_rmse(const SubModel &) const = 0;
    virtual void update(const SubModel &model) { noise->update(model); }
    virtual void get_pnm(const SubModel &,int,int,VectorNd &, MatrixNNd &) = 0;
    virtual void update_pnm(const SubModel &,int) = 0;

    //-- print info
    virtual std::ostream &info(std::ostream &os, std::string indent);

    // set noise models
    FixedGaussianNoise &setPrecision(double p);
    AdaptiveGaussianNoise &setAdaptivePrecision(double sn_init, double sn_max);
    ProbitNoise &setProbit();

    // virtual functions data-related
    virtual int    nmode() const = 0;
    virtual int    nnz()   const = 0;
            int    size()  const { return dim().dot(); }
    virtual int    nna()   const = 0;
    virtual PVec   dim()  const = 0;
    virtual double sum()   const = 0;
            int dim(int m) const { return dim().at(m); }
    // for matrices (nmode() == 2)
    virtual int nrow()     const { return dim(1); }
    virtual int ncol()     const { return dim(0); }

    // mean & centering
    double cwise_mean = NAN, global_mean = NAN;
    double mean(int m, int c) const { assert(mean_computed); return mode_mean.at(m)(c); }
    virtual double compute_mode_mean(int,int) = 0;
            void compute_mode_mean();
    virtual double offset_to_mean(const PVec &pos) const = 0;

    virtual void setCenterMode(std::string c);
    enum { CENTER_INVALID = -10, CENTER_NONE = -3, CENTER_GLOBAL = -2, CENTER_VIEW = -1, CENTER_COLS = 0, CENTER_ROWS = 1}
    center_mode;

    // helper for predictions
    double predict(const PVec &pos, const SubModel &model) const;

    // name
    std::string                  name;

    // noise model for this dataset
    std::unique_ptr<INoiseModel> noise;

  protected:
    std::vector<Eigen::VectorXd>  mode_mean;
    bool mean_computed = false;
    bool centered = false;
};

struct MatrixData: public Data {
    int nmode() const override { return 2; }
    std::ostream &info(std::ostream &os, std::string indent) override;
};

struct MatricesData: public MatrixData {
    MatricesData() : total_dim(2) { name = "MatricesData"; }

    void init_base() override;
    void setCenterMode(std::string c) override;

    void center(double) override;
    double compute_mode_mean(int,int) override;
    double offset_to_mean(const PVec &pos) const override;

    // add data
    MatrixData &add(const PVec &, std::unique_ptr<MatrixData>);

    // helper functions for noise
    // but 
    double sumsq(const SubModel &) const override { assert(false); return NAN; }
    double var_total() const override { assert(false); return NAN; }
    double train_rmse(const SubModel &) const override;

    // update noise and precision/mean
    void get_pnm(const SubModel &,int,int,VectorNd &, MatrixNNd &) override;
    void update_pnm(const SubModel &model, int mode) override;
  
    //-- print info
    std::ostream &info(std::ostream &os, std::string indent) override;

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

    int nview(int mode) { return mode_dim.at(mode).size(); }
    int view(int mode, int pos) {
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
    void init_base() override;

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

    void init_base() override;
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
