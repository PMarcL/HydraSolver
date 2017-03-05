#include "Variable.h"

using namespace std;

namespace hydra {

	Variable::Variable(const string& name) : name(name) {
	}


	Variable::~Variable() {
	}

	std::string Variable::getName() const {
		return name;
	}

} // namespace hydra