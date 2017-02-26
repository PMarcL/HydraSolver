#pragma once

namespace hydra {

class Model {
public:
	Model(int);
	~Model();

	int getTest() const;
private:
	int test;
};

} // namespace hydra


