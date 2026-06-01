# Makefile for the TwoPhaseFlowSolver1D Project

# --- User-configurable Paths ---
# Please edit these paths to match your system's installation locations.
EIGEN_PATH         := /Users/vigneshramakrishnan/Desktop/heatsink_opt/c++/eigen
BOOST_INCLUDE_PATH := /opt/homebrew/opt/boost/include
BOOST_LIB_PATH     := /opt/homebrew/opt/boost/lib
JSON_PATH          := /Users/vigneshramakrishnan/Desktop/RPI/Research/ONR-project/
NANOFLANN_PATH     := /Users/vigneshramakrishnan/Desktop/heatsink_opt/c++/nanoflann/include

# HPC Libraries
MFEM_INCLUDE_PATH  := /Users/vigneshramakrishnan/Desktop/software/mfem_install/include
MFEM_LIB_PATH      := /Users/vigneshramakrishnan/Desktop/software/mfem_install/lib
HYPRE_INCLUDE_PATH := /Users/vigneshramakrishnan/Desktop/software/hypre_install/include
HYPRE_LIB_PATH     := /Users/vigneshramakrishnan/Desktop/software/hypre_install/lib
GSLIB_INCLUDE_PATH := /Users/vigneshramakrishnan/Desktop/software/gslib/build/include
GSLIB_LIB_PATH     := /Users/vigneshramakrishnan/Desktop/software/gslib/build/lib
METIS_INCLUDE_PATH := /Users/vigneshramakrishnan/Desktop/software/metis_install/include
METIS_LIB_PATH     := /Users/vigneshramakrishnan/Desktop/software/metis_install/lib

# --- Python Wrapper Configuration (ANACONDA) ---
CONDA_ENV_PATH := /Users/vigneshramakrishnan/anaconda3/envs/pyHeatsink

# 1. Point to the Python executable
PY_EXECUTABLE := $(CONDA_ENV_PATH)/bin/python

# 2. Point explicitly to the config script (Conda usually names it python3-config)
PY_CONFIG_SCRIPT := $(CONDA_ENV_PATH)/bin/python3-config

# 3. Dynamic Python Paths
NUMPY_INCLUDE_PATH := $(shell $(PY_EXECUTABLE) -c "import numpy; print(numpy.get_include())")
PYBIND11_INCLUDES  := $(shell $(PY_EXECUTABLE) -m pybind11 --includes)
PYTHON_INCLUDES    := -I$(shell $(PY_EXECUTABLE) -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIB_PATH    := $(shell $(PY_EXECUTABLE) -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PY_EXTENSION_SUFFIX := $(shell $(PY_EXECUTABLE) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# 4. FIX FOR LINKER ERRORS:
# Get exact version string (e.g., "python3.11") to link the specific library file.
PYTHON_VER := $(shell $(PY_EXECUTABLE) -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")

# Manually construct the flags. We use -l$(PYTHON_VER) to explicitly link libpython3.11.dylib
PYTHON_EMBED_FLAGS := -L$(PYTHON_LIB_PATH) -l$(PYTHON_VER) -Wl,-rpath,$(PYTHON_LIB_PATH)
PYTHON_LINK_FLAGS := -undefined dynamic_lookup

# --- Compiler and Flags ---
CXX := mpicxx
CC  := mpicc
LDFLAGS := -L$(BOOST_LIB_PATH) -L$(MFEM_LIB_PATH) -L$(HYPRE_LIB_PATH) -L$(GSLIB_LIB_PATH) -L$(METIS_LIB_PATH)

# Flags for the Python module build AND the test build.
# Includes all necessary paths for pybind11 and matplotlib-cpp.
PY_CXXFLAGS := -std=c++20 -O3 -Wall -fPIC \
               -I$(EIGEN_PATH) -I$(BOOST_INCLUDE_PATH) -I$(JSON_PATH) \
               -I$(NANOFLANN_PATH) -I./src \
               -I$(MFEM_INCLUDE_PATH) -I$(HYPRE_INCLUDE_PATH) -I$(GSLIB_INCLUDE_PATH) -I$(METIS_INCLUDE_PATH) \
               $(PYTHON_INCLUDES) $(PYBIND11_INCLUDES) -I$(NUMPY_INCLUDE_PATH)

# Flags specific to the C++ executable build (no Python includes needed)
APP_CXXFLAGS := -std=c++20 -O3 -Wall \
                -I$(EIGEN_PATH) -I$(BOOST_INCLUDE_PATH) -I$(JSON_PATH) \
                -I$(NANOFLANN_PATH) -I./src \
                -I$(MFEM_INCLUDE_PATH) -I$(HYPRE_INCLUDE_PATH) -I$(GSLIB_INCLUDE_PATH) -I$(METIS_INCLUDE_PATH)

# --- Project Files ---
# 1. The Wrapper File
WRAPPER_SRC := ./src/coupled/HeatSinkWrapper.cpp

C_SRCS   := $(wildcard src/utils/*.c)
CPP_SRCS := $(wildcard src/utils/*.cpp) 

# 4. The Output Target Name for the Python Module
PY_MODULE_NAME   := heatsink_solver
PY_MODULE_TARGET := $(PY_MODULE_NAME)$(PY_EXTENSION_SUFFIX)

# Test Suite
TEST_EXECUTABLE := test_runner
TEST_CASE_SRCS  := $(wildcard test/*.cpp)
# NOTE: Test objects reuse the '.o' suffix and will be compiled with PY_CXXFLAGS
TEST_OBJECTS    := $(filter-out $(PY_WRAPPER_SRC:.cpp=.o), $(WRAPPER_OBJECTS)) $(TEST_CASE_SRCS:.cpp=.o)

# --- Sandbox Configuration ---
SANDBOX_SRCS := $(wildcard sandbox/*.cpp)
# Strip the directory path so executables are created in the current directory
SANDBOX_EXES := $(patsubst sandbox/%.cpp, %, $(SANDBOX_SRCS))
CORE_APP_OBJECTS := $(CPP_SRCS:.cpp=.o_app) $(C_SRCS:.c=.o_app) 

# --- Build Rules ---
.PHONY: all
all: $(EXECUTABLE)

$(EXECUTABLE): $(APP_OBJECTS)
	@echo "Linking executable..."
	$(CXX) $(APP_OBJECTS) -o $@ $(LDFLAGS) -lgs -lmfem -lHYPRE -lmetis -framework Accelerate
	@echo "Build complete. Executable is '$(EXECUTABLE)'"

# --- 1. Python Module Build Rule ---
.PHONY: module
module: $(PY_MODULE_TARGET)

$(PY_MODULE_TARGET): $(WRAPPER_SRC) $(CORE_OBJECTS)
	@echo "Building Python Module: $@"
	$(CXX) $(PY_CXXFLAGS) -shared $^ -o $@ $(LDFLAGS) \
		$(PYTHON_LINK_FLAGS) \
		-lgs -lmfem -lHYPRE -lmetis -framework Accelerate
	@echo "Signing module (macOS)..."
	codesign --force --sign - $@
	@echo "Build successful! Import with: import $(PY_MODULE_NAME)"

.PHONY: test
test: $(TEST_EXECUTABLE)

$(TEST_EXECUTABLE): $(TEST_OBJECTS)
	@echo "Linking test runner..."
	$(CXX) $(TEST_OBJECTS) -o $@ $(LDFLAGS) \
		$(PYTHON_EMBED_FLAGS) \
		-lgs -lmfem \
		$(HYPRE_LIB_PATH)/libHYPRE.a \
		-lmetis -framework Accelerate
	@echo "Signing test runner..."
	codesign --force --sign - $@
	@echo "Test build complete. Test executable is '$(TEST_EXECUTABLE)'"

.PHONY: sandbox
sandbox: $(SANDBOX_EXES)

# Rule to build sandbox executables in the root directory
# Maps 'executable_name' -> 'sandbox/executable_name.cpp'
$(SANDBOX_EXES): % : sandbox/%.cpp $(CORE_APP_OBJECTS)
	@echo "Building sandbox executable with Python support: $@"
	$(CXX) $(PY_CXXFLAGS) $< $(CORE_APP_OBJECTS) -o $@ $(LDFLAGS) \
		$(PYTHON_EMBED_FLAGS) \
		-lgs -lmfem -lHYPRE -lmetis -framework Accelerate
	@echo "Signing sandbox executable..."
	codesign --force --sign - $@

# Compilation rules for .o files (used by Python wrapper and tests)
%.o: %.cpp
	@echo "Compiling for Python/Test: $<"
	$(CXX) $(PY_CXXFLAGS) -I./tests -c $< -o $@

%.o: %.c
	@echo "Compiling C source for Python/Test: $<"
	$(CC) $(PY_CXXFLAGS) -c $< -o $@

# Rules for C++ application objects (.o_app)
%.o_app: %.cpp
	@echo "Compiling for C++ app: $<"
	$(CXX) $(APP_CXXFLAGS) -c $< -o $@

%.o_app: %.c
	@echo "Compiling C source for C++ app: $<"
	$(CC) $(APP_CXXFLAGS) -c $< -o $@

.PHONY: run-test
run-test: test
	@echo "Running tests..."
	./$(TEST_EXECUTABLE)

.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -f $(EXECUTABLE) $(TEST_EXECUTABLE) $(PY_MODULE) *.o *.o_app src/utils/*.o src/flow/*.o test/*.o *.csv *.pdf
	rm -f $(SANDBOX_EXES)
	rm -f $(PY_MODULE_TARGET)