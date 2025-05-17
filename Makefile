# Usage:
# make        # compile all binary
# make clean  # remove ALL binaries and objects

#
# A very simple makefile
#

# The default C++ compiler
# CC = g++ # serial version
CC=mpiCC

# The CFLAGS variable sets compile flags for gcc:
#  -g          compile with debug information
#  -Wall       give verbose compiler warnings
#  -O0         do not optimize generated code
#  -std=gnu99  use the GNU99 standard language definition
# INCLUDES =-I/home/peng276/.conda/envs/cent7/5.1.0-py36/mypackages/include
# LIBINCLUDES = -L/home/peng276/.conda/envs/cent7/5.1.0-py36/mypackages/lib
# INC = $(INCLUDES) $(LIBINCLUDES)


#CFLAGS = -g -pg -A -Wall -fno-inline # this is compile flag for debugging and profiling
CFLAGS = -Wall -fno-inline # this is compile flag for excution

# The LDFLAGS variable sets flags for linker
#  -lm   says to link in libm (the math library)
# LDFLAGS =  -lfmt -lm

# In this section, you list the files that are part of the project.
# If you add/change names of source files, here is where you
# edit the Makefile.
SOURCES = anBTE_mpi.cpp BTE_functs.cpp PhononClass.cpp Input.cpp
OBJECTS = $(SOURCES)
TARGET = anBTE_mpi.exe


# The first target defined in the makefile is the one
# used when make is invoked with no argument. Given the definitions
# above, this Makefile file will build the one named TARGET and
# assume that it depends on all the named OBJECTS files.

$(TARGET) : $(OBJECTS)
	$(CC) $(CFLAGS) $(INC) -o $@ $^ $(LDFLAGS)

.PHONY: clean

clean:
	rm -f $(TARGET) core
