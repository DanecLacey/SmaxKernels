#include "error_handler.hpp"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#define LINE_COUNT_WARNING 100000
#define LINE_COUNT_ERROR 1000000
#define BUF_SIZE 1024

namespace SMAX {
std::ofstream ErrorHandler::log_file;
int n_lines = 0;

std::string ErrorHandler::get_current_time() {
    // Get current time
    auto now = std::chrono::system_clock::now();

    // Get the time as time_t and as a tm structure
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_tm = *std::localtime(&now_time_t);

    // Get the microseconds part (fraction of the second)
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
                      now.time_since_epoch()) %
                  1000000;

    // Format time as [YYYY-MM-DD HH:MM:SS.mmmmmm]
    std::ostringstream oss;
    oss << "[" << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S") << "."
        << std::setw(6) << std::setfill('0') << micros.count() << "] ";

    return oss.str();
}

void ErrorHandler::_log(const std::string &log_message) {
    if (log_file.is_open()) {
        log_file << get_current_time() + log_message
                 << std::endl; // Write log to file
        ++n_lines;

        // Some protection against blowing up the file system
        if (n_lines % LINE_COUNT_WARNING == 0) {
            std::cerr << "Warning: " << LINE_COUNT_WARNING
                      << " lines in log file."
                      << " Consider turning off DEBUG_MODE." << std::endl;
        }
        if (n_lines == LINE_COUNT_ERROR) {
            std::cerr << "Error: " << LINE_COUNT_ERROR << " lines in log file."
                      << " Turn off DEBUG_MODE. Aborting." << std::endl;
            close_log();
            exit(EXIT_FAILURE);
        }
    } else {
        std::cerr << "Error: Unable to open log file!" << std::endl;
    }
}

void ErrorHandler::initialize_log(const std::string &filename) {
    log_file.open(filename, std::ios::out);
    if (!log_file.is_open()) {
        throw std::runtime_error("Failed to open log file: " + filename);
    }
}

void ErrorHandler::close_log() {
    if (log_file.is_open()) {
        log_file.close();
    }
}

void ErrorHandler::fatal(const std::string &message) {
    std::string full_message = "[FATAL] " + message;
    _log(full_message);
    throw std::runtime_error(full_message);
}

void ErrorHandler::warning(const std::string &message) {
    std::string full_message = "[WARNING] " + message;
    std::cerr << full_message << std::endl;
    _log(full_message);
}

void ErrorHandler::log(const std::string &message) { _log(message); }

void ErrorHandler::log(const char *format, ...) {

    char buffer[BUF_SIZE];

    va_list args;
    va_start(args, format);
    vsnprintf(buffer, BUF_SIZE, format, args);
    va_end(args);

    _log(std::string(buffer));
}

void ErrorHandler::kernel_dne(const std::string &kernel_name) {
    std::ostringstream oss;
    oss << "Kernel: '" << kernel_name << "' does not exist.";

    ErrorHandler::fatal(oss.str());
}

} // namespace SMAX