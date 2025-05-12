#ifndef SMAX_ERROR_HANDLER_HPP
#define SMAX_ERROR_HANDLER_HPP

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace SMAX {

class ErrorHandler {
  private:
    static std::ofstream log_file;
    static std::string get_current_time();
    static void _log(const std::string &log_message);

  public:
    static void initialize_log(const std::string &filename);
    static void close_log();
    static void fatal(const std::string &message);
    static void warning(const std::string &message);
    static void log(const std::string &message);
    static void log(const char *format, ...);
};

} // namespace SMAX

#endif // SMAX_ERROR_HANDLER_HPP