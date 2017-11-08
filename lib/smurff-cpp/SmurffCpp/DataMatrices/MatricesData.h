#pragma once

#include "MatrixData.h"

namespace smurff
{
   class MatricesData: public MatrixData
   {
   public:
      MatricesData();

   public:
      void init_pre() override;
      void init_post() override;
      
      // add data
      std::shared_ptr<MatrixData> add(const PVec<>& p, std::shared_ptr<MatrixData> data);

      // helper functions for noise
      // but
      double sumsq(const SubModel& model) const override;
      double var_total() const override;
      double train_rmse(const SubModel& model) const override;

      // update noise and precision/mean
      void update(const SubModel& model) override;
      void get_pnm(const SubModel& model, int mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM) override;
      void update_pnm(const SubModel& model, int mode) override;

      //-- print info
      std::ostream& info(std::ostream& os, std::string indent) override;
      std::ostream& status(std::ostream& os, std::string indent) const override;

      // accumulate on data in a block
      template<typename T, typename F>
      T accumulate(T init, F func) const
      {
         return std::accumulate(blocks.begin(), blocks.end(), init,
            [func](T s, const Block &b) -> T { return  s + (b.data().get()->*func)(); });
      }

      int    nnz() const override;
      int    nna() const override;
      double sum() const override;
      PVec<>   dim() const override;

   private:
      struct Block 
      {
          friend class MatricesData;
          // c'tor
          Block(PVec<> p, std::shared_ptr<MatrixData> m);

          // handy position functions
          const PVec<> start() const;
          const PVec<> end() const;
          const PVec<> dim() const;
          const PVec<> pos()  const;

          int start(int mode) const;
          int end(int mode) const;
          int dim(int mode) const;
          int pos(int mode) const;

          std::shared_ptr<MatrixData> data() const;

          bool in(const PVec<> &p) const;
          bool in(int mode, int p) const;

          SubModel submodel(const SubModel &model) const;

        private:
          PVec<> _pos, _start;
          std::shared_ptr<MatrixData> m_matrix;
      };
      std::vector<Block> blocks;

      template<typename Func>
      void apply(int mode, int p, Func f) const
      {
         for(auto &b : blocks) if (b.in(mode, p)) f(b);
      }

      const Block& find(const PVec<> &p) const;

      int nview(int mode) const override;
      int view(int mode, int pos) const override;
      int view_size(int mode, int v) const override;

      std::vector<std::vector<int>> mode_dim;
      PVec<> total_dim;
   };
}
