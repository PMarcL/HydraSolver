#pragma once

#include "HydraException.h"

namespace hydra {

	class IllegalVariableOperationException : public HydraException {
	public:
		IllegalVariableOperationException();

	private:
		static const std::string defaultCause;
	};

} // namesapce hydra
