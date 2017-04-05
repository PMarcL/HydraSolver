/******************************************************************************
Code adapted from

File: alldiff.cpp

Implementation of the algorithm for bounds consistency of the alldifferent
constraint from:

A. Lopez-Ortiz, C.-G. Quimper, J. Tromp, and P.  van Beek.
A fast and simple algorithm for bounds consistency of the
alldifferent constraint. IJCAI-2003.

By: John Tromp
******************************************************************************/

#include "AllDiffBoundsFilter.h"
#include "Variable.h"
#include <unordered_set>

using namespace std;

namespace hydra {

	const int INCONSISTENT = 0;
	const int CHANGES = 1;
	const int NO_CHANGES = 2;

	AllDiffBoundsFilter::AllDiffBoundsFilter(vector<Variable*> variables) : _vars(variables) {
		int i;

		n = variables.size();

		iv = (interval  *)calloc(n, sizeof(interval));
		minsorted = (interval **)calloc(n, sizeof(interval *));
		maxsorted = (interval **)calloc(n, sizeof(interval *));
		bounds = (int *)calloc(2 * n + 2, sizeof(int));

		for (i = 0; i < n; i++) {
			minsorted[i] = maxsorted[i] = &iv[i];
		}

		t = (int *)calloc(2 * n + 2, sizeof(int));
		d = (int *)calloc(2 * n + 2, sizeof(int));
		h = (int *)calloc(2 * n + 2, sizeof(int));
	}

	AllDiffBoundsFilter::~AllDiffBoundsFilter() {
		free(bounds);
		free(maxsorted);
		free(minsorted);
		free(iv);
		free(h);
		free(d);
		free(t);
	}

	void AllDiffBoundsFilter::replaceVariable(Variable* varToReplace, Variable* replacement) {
		for (size_t i = 0; i < _vars.size(); i++) {
			if (_vars[i] == varToReplace) {
				_vars[i] = replacement;
				break;
			}
		}
	}

	void sortmin(interval *v[], int n) {
		int i, current;
		bool sorted;
		interval *t;

		current = n - 1;
		sorted = false;
		while (!sorted) {
			sorted = true;
			for (i = 0; i < current; i++) {
				if (v[i]->min > v[i + 1]->min) {
					t = v[i];
					v[i] = v[i + 1];
					v[i + 1] = t;
					sorted = false;
				}
			}
			current--;
		}
	}

	void sortmax(interval *v[], int n) {
		int i, current;
		bool sorted;
		interval *t;

		current = 0;
		sorted = false;
		while (!sorted) {
			sorted = true;
			for (i = n - 1; i > current; i--) {
				if (v[i]->max < v[i - 1]->max) {
					t = v[i];
					v[i] = v[i - 1];
					v[i - 1] = t;
					sorted = false;
				}
			}
			current++;
		}
	}

	void AllDiffBoundsFilter::sortit() {
		int i, j, nb, min, max, last;

		sortmin(minsorted, n);
		sortmax(maxsorted, n);

		min = minsorted[0]->min;
		max = maxsorted[0]->max + 1;
		bounds[0] = last = min - 2;

		for (i = j = nb = 0;;) { // merge minsorted[] and maxsorted[] into bounds[]
			if (i < n && min <= max) {	// make sure minsorted exhausted first
				if (min != last)
					bounds[++nb] = last = min;
				minsorted[i]->minrank = nb;
				if (++i < n)
					min = minsorted[i]->min;
			} else {
				if (max != last)
					bounds[++nb] = last = max;
				maxsorted[j]->maxrank = nb;
				if (++j == n) break;
				max = maxsorted[j]->max + 1;
			}
		}
		AllDiffBoundsFilter::nb = nb;
		bounds[nb + 1] = bounds[nb] + 2;
	}

	void pathset(int *t, int start, int end, int to) {
		int k, l;
		for (l = start; (k = l) != end; t[k] = to) {
			l = t[k];
		}
	}

	int pathmin(int *t, int i) {
		for (; t[i] < i; i = t[i]) {
			;
		}
		return i;
	}

	int pathmax(int *t, int i) {
		for (; t[i] > i; i = t[i]) {
			;
		}
		return i;
	}

	int AllDiffBoundsFilter::filterlower() const {
		int i, j, w, x, y, z;
		auto changes = false;

		for (i = 1; i <= nb + 1; i++) {
			d[i] = bounds[i] - bounds[t[i] = h[i] = i - 1];
		}

		for (i = 0; i < n; i++) { // visit intervals in increasing max order
			x = maxsorted[i]->minrank; y = maxsorted[i]->maxrank;
			j = t[z = pathmax(t, x + 1)];

			if (--d[z] == 0) {
				t[z = pathmax(t, t[z] = z + 1)] = j;
			}
			pathset(t, x + 1, z, z); // path compression

			if (d[z] < bounds[z] - bounds[y]) {
				return INCONSISTENT; // no solution
			}

			if (h[x] > x) {
				maxsorted[i]->min = bounds[w = pathmax(h, h[x])];
				pathset(h, x, w, w); // path compression
				changes = true;
			}

			if (d[z] == bounds[z] - bounds[y]) {
				pathset(h, h[y], j - 1, y); // mark hall interval
				h[y] = j - 1; //("hall interval [%d,%d)\n",bounds[j],bounds[y]);
			}
		}

		if (changes) {
			return CHANGES;
		}
		return NO_CHANGES;
	}

	int AllDiffBoundsFilter::filterupper() const {
		int i, j, w, x, y, z;
		auto changes = false;

		for (i = 0; i <= nb; i++) {
			d[i] = bounds[t[i] = h[i] = i + 1] - bounds[i];
		}

		for (i = n; --i >= 0; ) { // visit intervals in decreasing min order
			x = minsorted[i]->maxrank; y = minsorted[i]->minrank;
			j = t[z = pathmin(t, x - 1)];
			if (--d[z] == 0) {
				t[z = pathmin(t, t[z] = z - 1)] = j;
			}
			pathset(t, x - 1, z, z);

			if (d[z] < bounds[y] - bounds[z]) {
				return INCONSISTENT; // no solution
			}

			if (h[x] < x) {
				minsorted[i]->max = bounds[w = pathmin(h, h[x])] - 1;
				pathset(h, x, w, w);
				changes = true;
			}

			if (d[z] == bounds[y] - bounds[z]) {
				pathset(h, h[y], j + 1, y);
				h[y] = j + 1;
			}
		}

		if (changes) {
			return CHANGES;
		}
		return NO_CHANGES;
	}

	bool AllDiffBoundsFilter::filter(vector<Variable*>& modifiedVariables) {
		auto i = 0;
		for (auto var : _vars) {
			iv[i].min = var->getLowerBound();
			iv[i].max = var->getUpperBound();
			i++;
		}

		sortit();


		auto status_upper = INCONSISTENT;

		auto status_lower = filterlower();
		if (status_lower != INCONSISTENT) {
			status_upper = filterupper();
		}

		if (status_lower == INCONSISTENT || status_upper == INCONSISTENT) {
			return false;
		}

		auto satisfied = true;

		for (auto var : _vars) {
			if (var->cardinality() <= 0) {
				satisfied = false;
				break;
			}
		}
		if (status_lower == CHANGES || status_upper == CHANGES) {
			unordered_set<Variable*> modifiedVariablesSet;
			i = 0;
			for (auto var : _vars) {
				auto modified = false;
				if (var->getLowerBound() < iv[i].min) {
					var->filterLowerBound(iv[i].min);
					modified = true;
				}

				if (var->getUpperBound() > iv[i].max) {
					var->filterUpperBound(iv[i].max);
					modified = true;
				}

				if (modified && modifiedVariablesSet.find(var) == modifiedVariablesSet.end()) {
					modifiedVariablesSet.insert(var);
				}

				satisfied = satisfied && var->cardinality() > 0;

				i++;
			}
			modifiedVariables.insert(modifiedVariables.begin(), modifiedVariablesSet.begin(), modifiedVariablesSet.end());
		}

		return satisfied;
	}
}
