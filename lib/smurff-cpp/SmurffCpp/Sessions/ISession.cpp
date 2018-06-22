#include <SmurffCpp/result.h>

#include "ISession.h"

using namespace smurff;

std::shared_ptr<std::vector<ResultItem>> ISession::getResultItems() const {
    return getResult()->m_predictions;
}
