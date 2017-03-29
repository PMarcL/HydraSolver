#include "TimeLogger.h"
#include <string>
#include <sstream>
#include <stdlib.h>

using namespace std;
using namespace chrono;

namespace hydra {

	TimeLogger::TimeLogger(string filename) : currentTime(high_resolution_clock::now()) {
		ostringstream os;
		os << filename << "_";
		os << rand() % 10000 << ".txt";
		file = ofstream(os.str());
	}

	TimeLogger::~TimeLogger() {
		for (auto time : times) {
			file << time << endl;
		}
		file.close();
	}

	void TimeLogger::tic() {
		currentTime = high_resolution_clock::now();
	}

	void TimeLogger::toc() {
		auto toc = high_resolution_clock::now();
		times.push_back(duration_cast<microseconds>(toc - currentTime).count());
	}

}
