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
CMAKE_SOURCE_DIR = /root/ascend_op/op/IsClose

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/ascend_op/op/IsClose/build_out

# Utility rule file for ascendc_bin_ascend310b_is_close_copy.

# Include any custom commands dependencies for this target.
include op_kernel/CMakeFiles/ascendc_bin_ascend310b_is_close_copy.dir/compiler_depend.make

# Include the progress variables for this target.
include op_kernel/CMakeFiles/ascendc_bin_ascend310b_is_close_copy.dir/progress.make

op_kernel/CMakeFiles/ascendc_bin_ascend310b_is_close_copy:
	cd /root/ascend_op/op/IsClose/build_out/op_kernel && cp /root/ascend_op/op/IsClose/build_out/op_kernel/tbe/dynamic/is_close.py /root/ascend_op/op/IsClose/build_out/op_kernel/binary/ascend310b/src/IsClose.py

ascendc_bin_ascend310b_is_close_copy: op_kernel/CMakeFiles/ascendc_bin_ascend310b_is_close_copy
ascendc_bin_ascend310b_is_close_copy: op_kernel/CMakeFiles/ascendc_bin_ascend310b_is_close_copy.dir/build.make
.PHONY : ascendc_bin_ascend310b_is_close_copy

# Rule to build all files generated by this target.
op_kernel/CMakeFiles/ascendc_bin_ascend310b_is_close_copy.dir/build: ascendc_bin_ascend310b_is_close_copy
.PHONY : op_kernel/CMakeFiles/ascendc_bin_ascend310b_is_close_copy.dir/build

op_kernel/CMakeFiles/ascendc_bin_ascend310b_is_close_copy.dir/clean:
	cd /root/ascend_op/op/IsClose/build_out/op_kernel && $(CMAKE_COMMAND) -P CMakeFiles/ascendc_bin_ascend310b_is_close_copy.dir/cmake_clean.cmake
.PHONY : op_kernel/CMakeFiles/ascendc_bin_ascend310b_is_close_copy.dir/clean

op_kernel/CMakeFiles/ascendc_bin_ascend310b_is_close_copy.dir/depend:
	cd /root/ascend_op/op/IsClose/build_out && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/ascend_op/op/IsClose /root/ascend_op/op/IsClose/op_kernel /root/ascend_op/op/IsClose/build_out /root/ascend_op/op/IsClose/build_out/op_kernel /root/ascend_op/op/IsClose/build_out/op_kernel/CMakeFiles/ascendc_bin_ascend310b_is_close_copy.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : op_kernel/CMakeFiles/ascendc_bin_ascend310b_is_close_copy.dir/depend

