#pragma once
#include <cstdlib>
#include <iostream>

// Debug-only logging.
#ifdef VDB_DEBUG
  #define DLOG(message) (std::cerr << "[dbg] " << message << std::endl)
#else
  #define DLOG(message) ((void)0)
#endif

// Always on unless you build with -DVDB_NO_ASSERT. Prints file, line, and a message.
#ifndef VDB_NO_ASSERT
  #define VDB_ASSERT(condition, message)                                     \
      do {                                                                   \
          if (!(condition)) {                                                \
              std::cerr << "ASSERT FAILED: " #condition "\n  " << message    \
                        << "\n  at " << __FILE__ << ":" << __LINE__          \
                        << " in " << __func__ << std::endl;                  \
              std::abort();                                                  \
          }                                                                  \
      } while (0)
#else
  #define VDB_ASSERT(condition, message) ((void)0)
#endif