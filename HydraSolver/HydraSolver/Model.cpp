#include "Model.h"



Model::Model(int t): test(t) {
}


Model::~Model() {
}

int Model::getTest() const {
	return test;
}
