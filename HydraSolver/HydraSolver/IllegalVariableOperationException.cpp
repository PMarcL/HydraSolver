#include "IllegalVariableOperationException.h"

using namespace std;

namespace hydra {

	const string IllegalVariableOperationException::defaultCause = "Illegal Variable Operation: An illegal operation was called on a variable";

	IllegalVariableOperationException::IllegalVariableOperationException() : HydraException(defaultCause) {
	}

} // namespace hydra
