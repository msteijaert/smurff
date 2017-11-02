#include "MatricesData.h"

using namespace smurff;

MatricesData::MatricesData()
   : total_dim(2)
{
   name = "MatricesData";
}

void MatricesData::init_pre()
{
   mode_dim.resize(nmode());
   for(int n = 0; n<nmode(); ++n)
   {
       std::vector<int> S(blocks.size());
       int max_pos = -1;
       for(auto &blk : blocks)
       {
           int pos  = blk.pos(n); // get coordinate of underlying matrix
           int size = blk.dim(n); // get dimension size of underlying matrix
           assert(size > 0);
           assert(S.at(pos) == 0 || S.at(pos) == size);
           S.at(pos) = size;
           max_pos = std::max(max_pos, pos);
       }
       int off = 0;
       auto &O = mode_dim.at(n);
       O.resize(max_pos+2);
       for(int pos=0; pos<=max_pos; ++pos)
       {
           O.at(pos) = off;
           off += S[pos];
       }
       O.at(max_pos+1) = off;
       total_dim.at(n) = off;
       for(auto &blk : blocks)
       {
           int pos = blk.pos(n);
           blk._start.at(n) = O[pos];
       }

   }

   init_cwise_mean();

   // init sub-matrices
   for(auto &p : blocks)
   {
       p.data()->init_pre();
       p.data()->compute_mode_mean();
   }
}

void MatricesData::init_post()
{
   Data::init_post();

   // init sub-matrices
   for(auto &p : blocks)
   {
      p.data()->init_post();
   }
}
void MatricesData::setCenterMode(std::string mode)
{
   Data::setCenterMode(mode);
   for(auto &p : blocks) 
      p.data()->setCenterMode(mode);
}

void MatricesData::setCenterMode(CenterModeTypes type)
{
   Data::setCenterMode(type);
   for(auto &p : blocks) 
      p.data()->setCenterMode(type);
}

void MatricesData::center(double global_mean)
{
    IMeanCentering::center(global_mean);

    // center sub-matrices
    assert(global_mean == getCwiseMean());

    for(auto &p : blocks)
      p.data()->center(getCwiseMean());

   setCentered(true);
}

double MatricesData::compute_mode_mean_mn(int mode, int pos)
{
   double sum = .0;
   int N = 0;
   int count = 0;

   apply(mode, pos, [&](const Block &b) {
       double local_mean = b.data()->getModeMeanItem(mode, pos - b.start(mode));
       sum += local_mean * b.dim(mode);
       N += b.dim(mode);
       count++;
   });

   assert(N>0);

   return sum / N;
}

double MatricesData::offset_to_mean(const PVec<>& pos) const
{
   const Block &b = find(pos);
   return b.data()->offset_to_mean(pos - b.start());
}

std::shared_ptr<MatrixData> MatricesData::add(const PVec<>& p, std::shared_ptr<MatrixData> data)
{
   blocks.push_back(Block(p, data));
   return blocks.back().data();
}

double MatricesData::sumsq(const SubModel& model) const
{
   assert(false);
   return NAN;
}

double MatricesData::var_total() const
{
   return NAN;
}

double MatricesData::train_rmse(const SubModel& model) const
{
   double sum = .0;
   int N = 0;
   int count = 0;

   for(auto &p : blocks)
   {
       auto mtx = p.data();
       double local_rmse = mtx->train_rmse(p.submodel(model));
       sum += (local_rmse * local_rmse) * (mtx->size() - mtx->nna());
       N += (mtx->size() - mtx->nna());
       count++;
   }

   assert(N>0);

   return sqrt(sum / N);
}

void MatricesData::update(const SubModel &model)
{
   for(auto &b : blocks)
   {
      b.data()->update(b.submodel(model));
   }
}

void MatricesData::get_pnm(const SubModel& model, int mode, int pos, Eigen::VectorXd& rr, Eigen::MatrixXd& MM)
{
   int count = 0;
   apply(mode, pos, [&model, mode, pos, &rr, &MM, &count](const Block &b) {
       b.data()->get_pnm(b.submodel(model), mode, pos - b.start(mode), rr, MM);
       count++;
   });
   assert(count>0);
}

void MatricesData::update_pnm(const SubModel& model, int m)
{
   for(auto &b : blocks) {
      b.data()->update_pnm(b.submodel(model), m);
  }
}

std::ostream& MatricesData::info(std::ostream& os, std::string indent)
{
   MatrixData::info(os, indent);
   os << indent << "Sub-Matrices:\n";
   for(auto &p : blocks)
   {
       os << indent;
       p.pos().info(os);
       os << ":\n";
       p.data()->info(os, indent + "  ");
       os << std::endl;
   }
   return os;
}

std::ostream& MatricesData::status(std::ostream& os, std::string indent) const
{
   os << indent << "Sub-Matrices:\n";
   for(auto &p : blocks)
   {
       os << indent << "  ";
       p.pos().info(os);
       os << ": " << p.data()->noise()->getStatus() << "\n";
   }
   return os;
}

int MatricesData::nnz() const
{
   return accumulate(0, &MatrixData::nnz);
}

int MatricesData::nna() const
{
   return accumulate(0, &MatrixData::nna);
}

double MatricesData::sum() const
{
   return accumulate(.0, &MatrixData::sum);
}

PVec<> MatricesData::dim() const
{
   return total_dim;
}

MatricesData::Block::Block(PVec<> p, std::shared_ptr<MatrixData> m)
   : _pos(p)
   , _start(2)
   , m_matrix(m)
{
}

const PVec<> MatricesData::Block::start() const
{
   return _start;
}

const PVec<> MatricesData::Block::end() const
{
   return start() + dim();
}

const PVec<> MatricesData::Block::dim() const
{
   return data()->dim();
}

const PVec<> MatricesData::Block::pos() const
{
   return _pos;
}

int MatricesData::Block::start(int mode) const
{
   return start().at(mode);
}

int MatricesData::Block::end(int mode) const
{
   return end().at(mode);
}

int MatricesData::Block::dim(int mode) const
{
   return dim().at(mode);
}

int MatricesData::Block::pos(int mode) const
{
   return pos().at(mode);
}

std::shared_ptr<MatrixData> MatricesData::Block::data() const
{
   return m_matrix;
}

bool MatricesData::Block::in(const PVec<> &p) const
{
   return p.in(start(), end());
}

bool MatricesData::Block::in(int mode, int p) const
{
   return p >= start(mode) && p < end(mode);
}

SubModel MatricesData::Block::submodel(const SubModel& model) const
{
   return SubModel(model, start(), dim());
}

const MatricesData::Block& MatricesData::find(const PVec<>& p) const
{
   return *std::find_if(blocks.begin(), blocks.end(), [p](const Block &b) -> bool { return b.in(p); });
}

int MatricesData::nview(int mode) const
{
   return mode_dim.at(mode).size() - 1;
}

int MatricesData::view(int mode, int pos) const
{
   assert(pos < MatrixData::dim(mode));
   const auto &v = mode_dim.at(mode);
   for(int i=0; i<nview(mode); ++i) if (pos < v.at(i + 1)) return i;
   assert(false);
   return -1;
}

int MatricesData::view_size(int mode, int v) const {
    const auto &M = mode_dim.at(mode);
    return M.at(v+1) - M.at(v);
}
