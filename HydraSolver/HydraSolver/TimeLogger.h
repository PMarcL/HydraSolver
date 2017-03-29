#pragma once

#include <vector>
#include <fstream>
#include <chrono>

namespace hydra {

	class TimeLogger {
	public:
		explicit TimeLogger(std::string filename);
		~TimeLogger();

		void tic();
		void toc();

	private:
		std::ofstream file;
		std::chrono::steady_clock::time_point currentTime;
		std::vector<long long> times;
	};

}
