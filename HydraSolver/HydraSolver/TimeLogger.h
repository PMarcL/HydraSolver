#pragma once

#include <vector>
#include <fstream>
#include <chrono>
#include <utility>

namespace hydra {

	class TimeLogger {
	public:
		explicit TimeLogger(std::string filename);
		~TimeLogger();

		void tic();
		void toc(int = 0);

	private:
		static int instances;

		std::ofstream file;
		std::chrono::steady_clock::time_point currentTime;
		std::vector<std::pair<long long, int>> times;
	};

}
