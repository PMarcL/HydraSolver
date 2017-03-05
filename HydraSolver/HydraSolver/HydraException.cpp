#include "HydraException.h"
#include <fstream>
#include <ctime>
#include <sstream>

using namespace std;

namespace hydra {

	const string HydraException::logFileExtension = ".txt";

	HydraException::HydraException(const string& what) : runtime_error(what) {
	}

	void HydraException::logException() const {
		ofstream ofs(getLogFileName());

		ofs << what() << endl;
		ofs << description << endl;

		ofs.close();
	}

	void HydraException::setDescription(const std::string& newDescription) {
		description = newDescription;
	}

	string HydraException::getLogFileName() {
		time_t rawTime;
		struct tm currentTime;

		time(&rawTime);
		localtime_s(&currentTime, &rawTime);

		ostringstream os;
		os << "Log_" << currentTime.tm_mon << "-" << currentTime.tm_mday << "-" << currentTime.tm_year << "_" <<
			currentTime.tm_hour << "_" << currentTime.tm_min << "_" << currentTime.tm_sec << logFileExtension;

		return os.str();
	}

} // namespace hydra
