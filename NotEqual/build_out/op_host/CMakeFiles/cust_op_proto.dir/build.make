# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/ascend_op/op/NotEqual

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/ascend_op/op/NotEqual/build_out

# Include any dependencies generated for this target.
include op_host/CMakeFiles/cust_op_proto.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include op_host/CMakeFiles/cust_op_proto.dir/compiler_depend.make

# Include the progress variables for this target.
include op_host/CMakeFiles/cust_op_proto.dir/progress.make

# Include the compile flags for this target's objects.
include op_host/CMakeFiles/cust_op_proto.dir/flags.make

op_host/CMakeFiles/cust_op_proto.dir/not_equal.cpp.o: op_host/CMakeFiles/cust_op_proto.dir/flags.make
op_host/CMakeFiles/cust_op_proto.dir/not_equal.cpp.o: ../op_host/not_equal.cpp
op_host/CMakeFiles/cust_op_proto.dir/not_equal.cpp.o: op_host/CMakeFiles/cust_op_proto.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/ascend_op/op/NotEqual/build_out/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object op_host/CMakeFiles/cust_op_proto.dir/not_equal.cpp.o"
	cd /root/ascend_op/op/NotEqual/build_out/op_host && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT op_host/CMakeFiles/cust_op_proto.dir/not_equal.cpp.o -MF CMakeFiles/cust_op_proto.dir/not_equal.cpp.o.d -o CMakeFiles/cust_op_proto.dir/not_equal.cpp.o -c /root/ascend_op/op/NotEqual/op_host/not_equal.cpp

op_host/CMakeFiles/cust_op_proto.dir/not_equal.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cust_op_proto.dir/not_equal.cpp.i"
	cd /root/ascend_op/op/NotEqual/build_out/op_host && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/ascend_op/op/NotEqual/op_host/not_equal.cpp > CMakeFiles/cust_op_proto.dir/not_equal.cpp.i

op_host/CMakeFiles/cust_op_proto.dir/not_equal.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cust_op_proto.dir/not_equal.cpp.s"
	cd /root/ascend_op/op/NotEqual/build_out/op_host && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/ascend_op/op/NotEqual/op_host/not_equal.cpp -o CMakeFiles/cust_op_proto.dir/not_equal.cpp.s

op_host/CMakeFiles/cust_op_proto.dir/__/autogen/op_proto.cc.o: op_host/CMakeFiles/cust_op_proto.dir/flags.make
op_host/CMakeFiles/cust_op_proto.dir/__/autogen/op_proto.cc.o: autogen/op_proto.cc
op_host/CMakeFiles/cust_op_proto.dir/__/autogen/op_proto.cc.o: op_host/CMakeFiles/cust_op_proto.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/ascend_op/op/NotEqual/build_out/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object op_host/CMakeFiles/cust_op_proto.dir/__/autogen/op_proto.cc.o"
	cd /root/ascend_op/op/NotEqual/build_out/op_host && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT op_host/CMakeFiles/cust_op_proto.dir/__/autogen/op_proto.cc.o -MF CMakeFiles/cust_op_proto.dir/__/autogen/op_proto.cc.o.d -o CMakeFiles/cust_op_proto.dir/__/autogen/op_proto.cc.o -c /root/ascend_op/op/NotEqual/build_out/autogen/op_proto.cc

op_host/CMakeFiles/cust_op_proto.dir/__/autogen/op_proto.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cust_op_proto.dir/__/autogen/op_proto.cc.i"
	cd /root/ascend_op/op/NotEqual/build_out/op_host && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/ascend_op/op/NotEqual/build_out/autogen/op_proto.cc > CMakeFiles/cust_op_proto.dir/__/autogen/op_proto.cc.i

op_host/CMakeFiles/cust_op_proto.dir/__/autogen/op_proto.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cust_op_proto.dir/__/autogen/op_proto.cc.s"
	cd /root/ascend_op/op/NotEqual/build_out/op_host && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/ascend_op/op/NotEqual/build_out/autogen/op_proto.cc -o CMakeFiles/cust_op_proto.dir/__/autogen/op_proto.cc.s

# Object files for target cust_op_proto
cust_op_proto_OBJECTS = \
"CMakeFiles/cust_op_proto.dir/not_equal.cpp.o" \
"CMakeFiles/cust_op_proto.dir/__/autogen/op_proto.cc.o"

# External object files for target cust_op_proto
cust_op_proto_EXTERNAL_OBJECTS =

op_host/libcust_opsproto_rt2.0.so: op_host/CMakeFiles/cust_op_proto.dir/not_equal.cpp.o
op_host/libcust_opsproto_rt2.0.so: op_host/CMakeFiles/cust_op_proto.dir/__/autogen/op_proto.cc.o
op_host/libcust_opsproto_rt2.0.so: op_host/CMakeFiles/cust_op_proto.dir/build.make
op_host/libcust_opsproto_rt2.0.so: op_host/CMakeFiles/cust_op_proto.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/ascend_op/op/NotEqual/build_out/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library libcust_opsproto_rt2.0.so"
	cd /root/ascend_op/op/NotEqual/build_out/op_host && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cust_op_proto.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
op_host/CMakeFiles/cust_op_proto.dir/build: op_host/libcust_opsproto_rt2.0.so
.PHONY : op_host/CMakeFiles/cust_op_proto.dir/build

op_host/CMakeFiles/cust_op_proto.dir/clean:
	cd /root/ascend_op/op/NotEqual/build_out/op_host && $(CMAKE_COMMAND) -P CMakeFiles/cust_op_proto.dir/cmake_clean.cmake
.PHONY : op_host/CMakeFiles/cust_op_proto.dir/clean

op_host/CMakeFiles/cust_op_proto.dir/depend:
	cd /root/ascend_op/op/NotEqual/build_out && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/ascend_op/op/NotEqual /root/ascend_op/op/NotEqual/op_host /root/ascend_op/op/NotEqual/build_out /root/ascend_op/op/NotEqual/build_out/op_host /root/ascend_op/op/NotEqual/build_out/op_host/CMakeFiles/cust_op_proto.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : op_host/CMakeFiles/cust_op_proto.dir/depend

