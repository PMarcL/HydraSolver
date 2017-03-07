#include "Variable.h"
#include "VariableObserver.h"

using namespace std;

namespace hydra {

	Variable::Variable(const string& name) : name(name) {
	}

	Variable::~Variable() {
	}

	string Variable::getName() const {
		return name;
	}

	void Variable::addObserver(VariableObserver* observer) {
		observers.push_back(observer);
	}

	void Variable::notifyDomainChanged() const {
		for (auto observer : observers) {
			observer->domainChanged();
		}
	}

} // namespace hydra