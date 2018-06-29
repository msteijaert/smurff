#include <SmurffCpp/result.h>

#include "ISession.h"

using namespace smurff;

const std::vector<ResultItem>& ISession::getResultItems() const {
    return getResult()->m_predictions;
}
