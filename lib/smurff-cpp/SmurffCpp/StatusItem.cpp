#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include "StatusItem.h"

using namespace smurff;

std::string StatusItem::asString() const
{
    std::ostringstream output;
    output << phase << " " << std::setfill(' ') << std::setw(3) << iter << "/" << std::setfill(' ') << std::setw(3) << phase_iter
           << ": RMSE: " << std::fixed << std::setprecision(4) << rmse_avg << " (1samp: " << std::fixed << std::setprecision(4) << rmse_1sample << ")";

    if (auc_1sample >= 0.0)
    {
        output << " AUC:" << std::fixed << std::setprecision(4) << auc_avg << " (1samp: " << std::fixed << std::setprecision(4) << auc_1sample << ")";
    }

    output << "  U:[";
    for (float n : model_norms)
    {
        output << std::scientific << std::setprecision(2) << n << ", ";
    }
    output << "] [took: " << std::fixed << std::setprecision(1) << elapsed_iter << "s, ";
    output << "total: " << std::fixed << std::setprecision(1) << elapsed_total << "s]";

    return output.str();
}