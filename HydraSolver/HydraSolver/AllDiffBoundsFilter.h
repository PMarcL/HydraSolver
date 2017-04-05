/******************************************************************************
Code adapted from

File: alldiff.h

Implementation of the algorithm for bounds consistency of the alldifferent
constraint from:

A. Lopez-Ortiz, C.-G. Quimper, J. Tromp, and P.  van Beek.
A fast and simple algorithm for bounds consistency of the
alldifferent constraint. IJCAI-2003.

By: John Tromp
******************************************************************************/

#pragma once

#include <vector>

namespace hydra {

	class Variable;

	typedef struct {
		int min, max; // start, end of interval
		int minrank, maxrank; // rank of min & max in bounds[] of an adcsp
	} interval;

	class AllDiffBoundsFilter {
	public:
		explicit AllDiffBoundsFilter(std::vector<Variable*>);
		~AllDiffBoundsFilter();

		bool filter(std::vector<Variable*>& modifiedVariables);
		void replaceVariable(Variable* varToReplace, Variable* replacement);

	private:
		void sortit();
		int filterlower() const;
		int filterupper() const;

		std::vector<Variable*> _vars;
		size_t n;
		int *t;		// tree links
		int *d;		// diffs between critical capacities
		int *h;		// hall interval links
		interval *iv;
		interval **minsorted;
		interval **maxsorted;
		int *bounds;  // bounds[1..nb] hold set of min & max in the niv intervals
					  // while bounds[0] and bounds[nb+1] allow sentinels
		int nb;
	};

}
