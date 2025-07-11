#include "error_handler.hpp"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#define SMAX_LINE_COUNT_WARNING 100000
#define SMAX_LINE_COUNT_ERROR 1000000
#define SMAX_BUF_SIZE 1024

namespace SMAX {
std::ofstream ErrorHandler::log_file;
int n_lines = 0;

std::string ErrorHandler::get_current_time() {
    // Get current time
    auto now = std::chrono::system_clock::now();

    // Convert to time_t
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);

    // Thread-safe conversion to tm using localtime_r
    std::tm now_tm;
    if (localtime_r(&now_time_t, &now_tm) == nullptr) {
        return "[TIME_ERROR] "; // Fallback in case of failure
    }

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
#pragma omp master
    {
        if (log_file.is_open()) {
            log_file << get_current_time() + log_message
                     << std::endl; // Write log to file
            ++n_lines;

            // Some protection against blowing up the file system
            if (n_lines % SMAX_LINE_COUNT_WARNING == 0) {
                std::cerr << "Warning: " << SMAX_LINE_COUNT_WARNING
                          << " lines in log file."
                          << " Consider turning off DEBUG_MODE." << std::endl;
            }
            if (n_lines == SMAX_LINE_COUNT_ERROR) {
                std::cerr << "Error: " << SMAX_LINE_COUNT_ERROR
                          << " lines in log file."
                          << " Turn off DEBUG_MODE. Aborting." << std::endl;
                close_log();
                exit(EXIT_FAILURE);
            }
        } else {
            std::cerr << "Error: Unable to open log file!" << std::endl;
        }
    }
}

void ErrorHandler::initialize_log(const std::string &filename) {
#pragma omp master
    {
        log_file.open(filename, std::ios::out);
        if (!log_file.is_open()) {
            throw std::runtime_error("Failed to open log file: " + filename);
        }
    }
} // namespace SMAX

void ErrorHandler::close_log() {
#pragma omp master
    {
        if (log_file.is_open()) {
            log_file.close();
        }
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

    char buffer[SMAX_BUF_SIZE];

    va_list args;
    va_start(args, format);
    vsnprintf(buffer, SMAX_BUF_SIZE, format, args);
    va_end(args);

    _log(std::string(buffer));
}

void ErrorHandler::kernel_dne(const std::string &kernel_name) {
    std::ostringstream oss;
    oss << "Kernel: '" << kernel_name << "' does not exist.";

    ErrorHandler::fatal(oss.str());
}

} // namespace SMAX