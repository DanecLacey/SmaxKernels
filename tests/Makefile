include config.mk

# File paths
INSTALLED_LIB=${INSTALL_PATH}/lib
HEADERS=${INSTALL_PATH}/include

# Set default SmaxKernel links
PATH_EMBEDDINGS+=-Wl,-rpath=${INSTALLED_LIB}
LINKS+=-lsmaxkernels

CXXFLAGS=-g -Wall -Wextra -std=c++14

all: test1 test2

test1: 
	${CXX} ${CXXFLAGS} test1.cpp -o $@ ${LIBS} -I${HEADERS} -L${INSTALLED_LIB} ${LINKS} ${PATH_EMBEDDINGS}

test2: 
	${CXX} ${CXXFLAGS} test2.cpp -o $@ ${LIBS} -I${HEADERS} -L${INSTALLED_LIB} ${LINKS} ${PATH_EMBEDDINGS}

clean:
	rm -f test1 test2 test3
	rm -f *.o

.PHONY : clean