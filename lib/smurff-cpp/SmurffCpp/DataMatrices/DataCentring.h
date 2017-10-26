#pragma once

class DataCentring;

// mean and centering
class IMeanCenteringOld
{
private:
   double m_cwise_mean = NAN; // mean of non NA elements in matrix
   bool m_cwise_mean_initialized = false;
   double m_global_mean = NAN;
   double m_var = NAN;
   CenterModeTypes m_center_mode;

private:
   std::vector<Eigen::VectorXd> m_mode_mean;
   bool m_mean_computed = false;
   bool m_centered = false;

protected:
   IMeanCentering();

public:
   virtual ~IMeanCentering() {}

   //AGE: methods are pure virtual because they depend on multiple interfaces from Data class
   //introducing Data class dependency into this class is not a good idea
protected:
   void compute_mode_mean_internal(const Data* data); //depends on cwise mean. Also depends on initial data from Ycentered (which is Y)
   virtual void compute_mode_mean() = 0;
   virtual double compute_mode_mean_mn(int mode, int pos) = 0;

   void init_cwise_mean_internal(const Data* data); //does not depend on any init
   virtual void init_cwise_mean() = 0;  //does not depend on any init

   //AGE: implementation depends on matrix data type
protected:
   virtual void center(double upper_mean); //depends on cwise mean. Also depends on mod mean being calculated.

public:
   virtual void setCenterMode(std::string c);
   virtual void setCenterMode(smurff::CenterModeTypes type);

   //AGE: implementation depends on matrix data type
public:
   virtual double offset_to_mean(const PVec<>& pos) const = 0;

   //AGE: getters
public:
   double getCwiseMean() const;
   double getGlobalMean() const;
   double getVar() const;
   smurff::CenterModeTypes getCenterMode() const;
   bool getMeanComputed() const;
   double getModeMeanItem(int m, int c) const;
   const Eigen::VectorXd& getModeMean(size_t i) const;
   std::string getCenterModeName() const;

public:
   void setCentered(bool value)
   {
      m_centered = value;
   }
};

class DataCentring : public IMeanCenteringOld
{
// #### mean centring functions  ####
public:
   DataCentring()
      : IMeanCenteringOld
   {

   }

   void compute_mode_mean() override;

   void init_cwise_mean() override;

   void init()
   {
      //compute global mean & mode-wise means
      compute_mode_mean();
      center(getCwiseMean());
   }
};

class MatrixDataCentring : public Data
{
};

class MatricesDataCentring: public MatrixDataCentring
{
public:
   void setCenterMode(std::string mode) override;
   void setCenterMode(CenterModeTypes type) override;

   void center(double global_mean) override;
   double compute_mode_mean_mn(int mode, int pos) override;
   double offset_to_mean(const PVec<>& pos) const override;

   void init_pre() override
   {
      init_cwise_mean();
      
      // init sub-matrices
      for(auto &p : blocks)
      {
            p.data().init_pre();
            p.data().compute_mode_mean();
      }
   }
};

template<typename YType>
class MatrixDataTemplCentring : public MatrixDataCentring
{
public:
   void init_pre() override
   {
      Ycentered = std::shared_ptr<std::vector<YType> >(new std::vector<YType>());
      Ycentered->push_back(Y.transpose());
      Ycentered->push_back(Y);

      init_cwise_mean();
   }

   double offset_to_mean(const PVec<>& pos) const override
   {
           if (getCenterMode() == CenterModeTypes::CENTER_GLOBAL) return getGlobalMean();
      else if (getCenterMode() == CenterModeTypes::CENTER_VIEW)   return getCwiseMean();
      else if (getCenterMode() == CenterModeTypes::CENTER_ROWS)   return getModeMeanItem(1,pos.at(1));
      else if (getCenterMode() == CenterModeTypes::CENTER_COLS)   return getModeMeanItem(0,pos.at(0));
      else if (getCenterMode() == CenterModeTypes::CENTER_NONE)   return .0;
      assert(false);
      return .0;
   }

private:
   std::shared_ptr<std::vector<YType> > Ycentered; // centered versions of original matrix (transposed, original)

public:
   const std::vector<YType>& getYc() const
   {
      assert(Ycentered);
      return *Ycentered.get();
   }

   std::shared_ptr<std::vector<YType> > getYcPtr() const
   {
      assert(Ycentered);
      return Ycentered;
   }
};

template<class YType>
class FullMatrixDataCentring : public MatrixDataTemplCentring<YType>
{
private:
   double compute_mode_mean_mn(int mode, int pos) override
   {
      const auto &col = this->getYcPtr()->at(mode).col(pos);
      if (col.nonZeros() == 0)
         return this->getCwiseMean();
      return col.sum() / this->getYcPtr()->at(mode).rows();
   }
};

class ScarceMatrixDataCentring : public MatrixDataTemplCentring<Eigen::SparseMatrix<double> >
{
public:
   void center(double global_mean) override;
   double compute_mode_mean_mn(int mode, int pos) override;
};

class DenseMatrixDataCentring : public FullMatrixDataCentring<Eigen::MatrixXd>
{
public:
   void center(double global_mean) override;
};

class SparseMatrixDataCentring : public FullMatrixDataCentring<Eigen::SparseMatrix<double> >
{
public:
   void center(double global_mean) override;
};
