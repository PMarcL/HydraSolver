#pragma once

#include <stdexcept>
#include <string>

namespace hydra {

	class HydraException : public std::runtime_error {
	public:
		explicit HydraException(const std::string&);

		void logException() const;
		void setDescription(const std::string&);

	private:
		static std::string getLogFileName();
		static const std::string logFileExtension;

		std::string description;
	};

} // namespace hydra

