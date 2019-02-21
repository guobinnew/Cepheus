# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/unique/Work/opencv

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/unique/Work/opencv

# Include any dependencies generated for this target.
include CMakeFiles/opencv_qrcode.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/opencv_qrcode.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/opencv_qrcode.dir/flags.make

CMakeFiles/opencv_qrcode.dir/qrcode.cpp.o: CMakeFiles/opencv_qrcode.dir/flags.make
CMakeFiles/opencv_qrcode.dir/qrcode.cpp.o: qrcode.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/unique/Work/opencv/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/opencv_qrcode.dir/qrcode.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opencv_qrcode.dir/qrcode.cpp.o -c /home/unique/Work/opencv/qrcode.cpp

CMakeFiles/opencv_qrcode.dir/qrcode.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_qrcode.dir/qrcode.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/unique/Work/opencv/qrcode.cpp > CMakeFiles/opencv_qrcode.dir/qrcode.cpp.i

CMakeFiles/opencv_qrcode.dir/qrcode.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_qrcode.dir/qrcode.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/unique/Work/opencv/qrcode.cpp -o CMakeFiles/opencv_qrcode.dir/qrcode.cpp.s

CMakeFiles/opencv_qrcode.dir/qrcode.cpp.o.requires:

.PHONY : CMakeFiles/opencv_qrcode.dir/qrcode.cpp.o.requires

CMakeFiles/opencv_qrcode.dir/qrcode.cpp.o.provides: CMakeFiles/opencv_qrcode.dir/qrcode.cpp.o.requires
	$(MAKE) -f CMakeFiles/opencv_qrcode.dir/build.make CMakeFiles/opencv_qrcode.dir/qrcode.cpp.o.provides.build
.PHONY : CMakeFiles/opencv_qrcode.dir/qrcode.cpp.o.provides

CMakeFiles/opencv_qrcode.dir/qrcode.cpp.o.provides.build: CMakeFiles/opencv_qrcode.dir/qrcode.cpp.o


# Object files for target opencv_qrcode
opencv_qrcode_OBJECTS = \
"CMakeFiles/opencv_qrcode.dir/qrcode.cpp.o"

# External object files for target opencv_qrcode
opencv_qrcode_EXTERNAL_OBJECTS =

opencv_qrcode: CMakeFiles/opencv_qrcode.dir/qrcode.cpp.o
opencv_qrcode: CMakeFiles/opencv_qrcode.dir/build.make
opencv_qrcode: /usr/local/opencv4/lib/libopencv_dnn.so.4.0.1
opencv_qrcode: /usr/local/opencv4/lib/libopencv_gapi.so.4.0.1
opencv_qrcode: /usr/local/opencv4/lib/libopencv_ml.so.4.0.1
opencv_qrcode: /usr/local/opencv4/lib/libopencv_objdetect.so.4.0.1
opencv_qrcode: /usr/local/opencv4/lib/libopencv_photo.so.4.0.1
opencv_qrcode: /usr/local/opencv4/lib/libopencv_stitching.so.4.0.1
opencv_qrcode: /usr/local/opencv4/lib/libopencv_video.so.4.0.1
opencv_qrcode: /usr/local/opencv4/lib/libopencv_calib3d.so.4.0.1
opencv_qrcode: /usr/local/opencv4/lib/libopencv_features2d.so.4.0.1
opencv_qrcode: /usr/local/opencv4/lib/libopencv_flann.so.4.0.1
opencv_qrcode: /usr/local/opencv4/lib/libopencv_highgui.so.4.0.1
opencv_qrcode: /usr/local/opencv4/lib/libopencv_videoio.so.4.0.1
opencv_qrcode: /usr/local/opencv4/lib/libopencv_imgcodecs.so.4.0.1
opencv_qrcode: /usr/local/opencv4/lib/libopencv_imgproc.so.4.0.1
opencv_qrcode: /usr/local/opencv4/lib/libopencv_core.so.4.0.1
opencv_qrcode: CMakeFiles/opencv_qrcode.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/unique/Work/opencv/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable opencv_qrcode"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/opencv_qrcode.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/opencv_qrcode.dir/build: opencv_qrcode

.PHONY : CMakeFiles/opencv_qrcode.dir/build

CMakeFiles/opencv_qrcode.dir/requires: CMakeFiles/opencv_qrcode.dir/qrcode.cpp.o.requires

.PHONY : CMakeFiles/opencv_qrcode.dir/requires

CMakeFiles/opencv_qrcode.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/opencv_qrcode.dir/cmake_clean.cmake
.PHONY : CMakeFiles/opencv_qrcode.dir/clean

CMakeFiles/opencv_qrcode.dir/depend:
	cd /home/unique/Work/opencv && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/unique/Work/opencv /home/unique/Work/opencv /home/unique/Work/opencv /home/unique/Work/opencv /home/unique/Work/opencv/CMakeFiles/opencv_qrcode.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/opencv_qrcode.dir/depend
