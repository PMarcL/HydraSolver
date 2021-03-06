#pragma once

namespace hydra {

	class IntVariableIterator {
	public:
		IntVariableIterator();
		virtual ~IntVariableIterator();

		virtual int next() = 0;
		virtual int previous() = 0;
		virtual bool hasNextValue() const = 0;
	};

} // namespace hydra
