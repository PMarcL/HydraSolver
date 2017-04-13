#include "TimeLogger.h"
#include <string>
#include <sstream>

using namespace std;
using namespace chrono;

namespace hydra {

	int TimeLogger::instances = 0;

	TimeLogger::TimeLogger(string filename) : currentTime(high_resolution_clock::now()) {
		instances++;
		ostringstream os;
		os << filename << "_";
		os << instances << ".txt";
		file = ofstream(os.str());
	}

	TimeLogger::~TimeLogger() {
		for (auto time : times) {
			file << time.first << " " << time.second << endl;
		}
		file.close();
	}

	void TimeLogger::tic() {
		currentTime = high_resolution_clock::now();
	}

	void TimeLogger::toc(int i) {
		auto toc = high_resolution_clock::now();
		times.push_back(pair<long long, int>(duration_cast<microseconds>(toc - currentTime).count(), i));
	}

}
