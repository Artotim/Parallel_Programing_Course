# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files (x86)\JetBrains\CLion 2020.1\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files (x86)\JetBrains\CLion 2020.1\bin\cmake\win\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\Tango\Desktop\Final_parallel

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\Tango\Desktop\Final_parallel\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/Final_parallel.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Final_parallel.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Final_parallel.dir/flags.make

CMakeFiles/Final_parallel.dir/main.c.obj: CMakeFiles/Final_parallel.dir/flags.make
CMakeFiles/Final_parallel.dir/main.c.obj: ../main.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\Tango\Desktop\Final_parallel\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/Final_parallel.dir/main.c.obj"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles\Final_parallel.dir\main.c.obj   -c C:\Users\Tango\Desktop\Final_parallel\main.c

CMakeFiles/Final_parallel.dir/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/Final_parallel.dir/main.c.i"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\Tango\Desktop\Final_parallel\main.c > CMakeFiles\Final_parallel.dir\main.c.i

CMakeFiles/Final_parallel.dir/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/Final_parallel.dir/main.c.s"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\Tango\Desktop\Final_parallel\main.c -o CMakeFiles\Final_parallel.dir\main.c.s

CMakeFiles/Final_parallel.dir/sparseKMeans.c.obj: CMakeFiles/Final_parallel.dir/flags.make
CMakeFiles/Final_parallel.dir/sparseKMeans.c.obj: ../sparseKMeans.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\Tango\Desktop\Final_parallel\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/Final_parallel.dir/sparseKMeans.c.obj"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles\Final_parallel.dir\sparseKMeans.c.obj   -c C:\Users\Tango\Desktop\Final_parallel\sparseKMeans.c

CMakeFiles/Final_parallel.dir/sparseKMeans.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/Final_parallel.dir/sparseKMeans.c.i"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\Tango\Desktop\Final_parallel\sparseKMeans.c > CMakeFiles\Final_parallel.dir\sparseKMeans.c.i

CMakeFiles/Final_parallel.dir/sparseKMeans.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/Final_parallel.dir/sparseKMeans.c.s"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\Tango\Desktop\Final_parallel\sparseKMeans.c -o CMakeFiles\Final_parallel.dir\sparseKMeans.c.s

CMakeFiles/Final_parallel.dir/updateCenters.c.obj: CMakeFiles/Final_parallel.dir/flags.make
CMakeFiles/Final_parallel.dir/updateCenters.c.obj: ../updateCenters.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\Tango\Desktop\Final_parallel\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/Final_parallel.dir/updateCenters.c.obj"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles\Final_parallel.dir\updateCenters.c.obj   -c C:\Users\Tango\Desktop\Final_parallel\updateCenters.c

CMakeFiles/Final_parallel.dir/updateCenters.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/Final_parallel.dir/updateCenters.c.i"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\Tango\Desktop\Final_parallel\updateCenters.c > CMakeFiles\Final_parallel.dir\updateCenters.c.i

CMakeFiles/Final_parallel.dir/updateCenters.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/Final_parallel.dir/updateCenters.c.s"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\Tango\Desktop\Final_parallel\updateCenters.c -o CMakeFiles\Final_parallel.dir\updateCenters.c.s

CMakeFiles/Final_parallel.dir/updateWeights.c.obj: CMakeFiles/Final_parallel.dir/flags.make
CMakeFiles/Final_parallel.dir/updateWeights.c.obj: ../updateWeights.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\Tango\Desktop\Final_parallel\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/Final_parallel.dir/updateWeights.c.obj"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles\Final_parallel.dir\updateWeights.c.obj   -c C:\Users\Tango\Desktop\Final_parallel\updateWeights.c

CMakeFiles/Final_parallel.dir/updateWeights.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/Final_parallel.dir/updateWeights.c.i"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\Tango\Desktop\Final_parallel\updateWeights.c > CMakeFiles\Final_parallel.dir\updateWeights.c.i

CMakeFiles/Final_parallel.dir/updateWeights.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/Final_parallel.dir/updateWeights.c.s"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\Tango\Desktop\Final_parallel\updateWeights.c -o CMakeFiles\Final_parallel.dir\updateWeights.c.s

CMakeFiles/Final_parallel.dir/minMaxKMeans.c.obj: CMakeFiles/Final_parallel.dir/flags.make
CMakeFiles/Final_parallel.dir/minMaxKMeans.c.obj: ../minMaxKMeans.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\Tango\Desktop\Final_parallel\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object CMakeFiles/Final_parallel.dir/minMaxKMeans.c.obj"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles\Final_parallel.dir\minMaxKMeans.c.obj   -c C:\Users\Tango\Desktop\Final_parallel\minMaxKMeans.c

CMakeFiles/Final_parallel.dir/minMaxKMeans.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/Final_parallel.dir/minMaxKMeans.c.i"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\Tango\Desktop\Final_parallel\minMaxKMeans.c > CMakeFiles\Final_parallel.dir\minMaxKMeans.c.i

CMakeFiles/Final_parallel.dir/minMaxKMeans.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/Final_parallel.dir/minMaxKMeans.c.s"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\Tango\Desktop\Final_parallel\minMaxKMeans.c -o CMakeFiles\Final_parallel.dir\minMaxKMeans.c.s

CMakeFiles/Final_parallel.dir/read_write_routines.c.obj: CMakeFiles/Final_parallel.dir/flags.make
CMakeFiles/Final_parallel.dir/read_write_routines.c.obj: ../read_write_routines.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\Tango\Desktop\Final_parallel\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object CMakeFiles/Final_parallel.dir/read_write_routines.c.obj"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles\Final_parallel.dir\read_write_routines.c.obj   -c C:\Users\Tango\Desktop\Final_parallel\read_write_routines.c

CMakeFiles/Final_parallel.dir/read_write_routines.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/Final_parallel.dir/read_write_routines.c.i"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\Tango\Desktop\Final_parallel\read_write_routines.c > CMakeFiles\Final_parallel.dir\read_write_routines.c.i

CMakeFiles/Final_parallel.dir/read_write_routines.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/Final_parallel.dir/read_write_routines.c.s"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\Tango\Desktop\Final_parallel\read_write_routines.c -o CMakeFiles\Final_parallel.dir\read_write_routines.c.s

CMakeFiles/Final_parallel.dir/array_routines.c.obj: CMakeFiles/Final_parallel.dir/flags.make
CMakeFiles/Final_parallel.dir/array_routines.c.obj: ../array_routines.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\Tango\Desktop\Final_parallel\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building C object CMakeFiles/Final_parallel.dir/array_routines.c.obj"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles\Final_parallel.dir\array_routines.c.obj   -c C:\Users\Tango\Desktop\Final_parallel\array_routines.c

CMakeFiles/Final_parallel.dir/array_routines.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/Final_parallel.dir/array_routines.c.i"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\Tango\Desktop\Final_parallel\array_routines.c > CMakeFiles\Final_parallel.dir\array_routines.c.i

CMakeFiles/Final_parallel.dir/array_routines.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/Final_parallel.dir/array_routines.c.s"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\Tango\Desktop\Final_parallel\array_routines.c -o CMakeFiles\Final_parallel.dir\array_routines.c.s

# Object files for target Final_parallel
Final_parallel_OBJECTS = \
"CMakeFiles/Final_parallel.dir/main.c.obj" \
"CMakeFiles/Final_parallel.dir/sparseKMeans.c.obj" \
"CMakeFiles/Final_parallel.dir/updateCenters.c.obj" \
"CMakeFiles/Final_parallel.dir/updateWeights.c.obj" \
"CMakeFiles/Final_parallel.dir/minMaxKMeans.c.obj" \
"CMakeFiles/Final_parallel.dir/read_write_routines.c.obj" \
"CMakeFiles/Final_parallel.dir/array_routines.c.obj"

# External object files for target Final_parallel
Final_parallel_EXTERNAL_OBJECTS =

Final_parallel.exe: CMakeFiles/Final_parallel.dir/main.c.obj
Final_parallel.exe: CMakeFiles/Final_parallel.dir/sparseKMeans.c.obj
Final_parallel.exe: CMakeFiles/Final_parallel.dir/updateCenters.c.obj
Final_parallel.exe: CMakeFiles/Final_parallel.dir/updateWeights.c.obj
Final_parallel.exe: CMakeFiles/Final_parallel.dir/minMaxKMeans.c.obj
Final_parallel.exe: CMakeFiles/Final_parallel.dir/read_write_routines.c.obj
Final_parallel.exe: CMakeFiles/Final_parallel.dir/array_routines.c.obj
Final_parallel.exe: CMakeFiles/Final_parallel.dir/build.make
Final_parallel.exe: CMakeFiles/Final_parallel.dir/linklibs.rsp
Final_parallel.exe: CMakeFiles/Final_parallel.dir/objects1.rsp
Final_parallel.exe: CMakeFiles/Final_parallel.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\Tango\Desktop\Final_parallel\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking C executable Final_parallel.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\Final_parallel.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Final_parallel.dir/build: Final_parallel.exe

.PHONY : CMakeFiles/Final_parallel.dir/build

CMakeFiles/Final_parallel.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\Final_parallel.dir\cmake_clean.cmake
.PHONY : CMakeFiles/Final_parallel.dir/clean

CMakeFiles/Final_parallel.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\Tango\Desktop\Final_parallel C:\Users\Tango\Desktop\Final_parallel C:\Users\Tango\Desktop\Final_parallel\cmake-build-debug C:\Users\Tango\Desktop\Final_parallel\cmake-build-debug C:\Users\Tango\Desktop\Final_parallel\cmake-build-debug\CMakeFiles\Final_parallel.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Final_parallel.dir/depend

