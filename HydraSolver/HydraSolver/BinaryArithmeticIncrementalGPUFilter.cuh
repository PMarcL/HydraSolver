#pragma once

#include <vector>

namespace hydra {

	class BitsetIntVariable;
	class Variable;

	class Variable;

	enum Operator {
		MINUS,
		PLUS,
		MULTIPLIES,
		DIVIDES
	};

	enum RelationalOperator {
		EQ,
		NEQ,
		GT,
		GEQ,
		LT,
		LEQ
	};

	class BinaryArithmeticIncrementalGPUFilter {
	public:
		BinaryArithmeticIncrementalGPUFilter(BitsetIntVariable* var1, BitsetIntVariable* var2, Operator op, RelationalOperator relop, int rhs);
		~BinaryArithmeticIncrementalGPUFilter();

		std::vector<Variable*> filterBoundsGPU();
		std::vector<Variable*> filterDomainGPU() const;

	private:
		void updateVar1DeviceAttributes();
		void updateVar2DeviceAttributes();
		bool filterVariableBounds(BitsetIntVariable* var, int *lb, int *ub, int *originalLowerBound, uint8_t *bitset_device, uint8_t *bitset_host) const;
		bool filterVariableDomain(BitsetIntVariable* var1, BitsetIntVariable* var2, int *var1OriginalLowerBound, int *var2OriginalLowerBound, unsigned int *var2Size,
			uint8_t *bitsetDeviceVar1, uint8_t *bitsetDeviceVar2, uint8_t *bitsetHostVar1, uint8_t *bitset_matrix, bool *varIsFirst) const;

		BitsetIntVariable *var1;
		BitsetIntVariable *var2;
		Operator op;
		RelationalOperator relop;
		int rhs;

		int var1_lb;
		int var1_ub;
		int var2_lb;
		int var2_ub;

		int *device_rhs;
		int *deviceVar1Original_lb;
		int *deviceVar2Original_lb;
		int *deviceVar1_lb;
		int *deviceVar1_ub;
		int *deviceVar2_lb;
		int *deviceVar2_ub;
		unsigned int *deviceVar1_size;
		unsigned int *deviceVar2_size;
		uint8_t * bitset_device_var1;
		uint8_t * bitset_device_var2;
		uint8_t * bitset_host_var1;
		uint8_t * bitset_host_var2;
		uint8_t * bitset_matrix_var1;
		uint8_t * bitset_matrix_var2;
	};

}