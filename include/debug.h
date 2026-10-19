#pragma once
#include <cstdlib>
#include <iostream>

// debug-only logging 
#ifdef VDB_DEBUG
  #define DLOG(x) (std::cerr << "[dbg] " << x << std::endl)
#else
  #define DLOG(x) ((void)0)
#endif


// Always on unless you build with -DVDB_NO_ASSERT. Prints file, line, and a message.
#ifndef VDB_NO_ASSERT
  #define VDB_ASSERT(cond, msg)                                              \
      do {                                                                   \
          if (!(cond)) {                                                     \
              std::cerr << "ASSERT FAILED: " #cond "\n  " << msg             \
                        << "\n  at " << __FILE__ << ":" << __LINE__          \
                        << " in " << __func__ << std::endl;                  \
              std::abort();                                                  \
          }                                                                  \
      } while (0)
#else
  #define VDB_ASSERT(cond, msg) ((void)0)
#endif