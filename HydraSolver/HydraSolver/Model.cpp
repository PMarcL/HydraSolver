#include "Model.h"

namespace hydra {

Model::Model(int t): test(t) {
}

Model::~Model() {
}

int Model::getTest() const {
	return test;
}

} // namespace hydra
