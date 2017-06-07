#Global Settings
KOKKOS_PATH = /Users/kehw/tmp/kokkos-master
KOKKOS_DEVICES = "Pthreads"
KOKKOS_ARCH = ""
EXE_NAME = "test"


#Source Files
SRC = $(wildcard *.cpp)

default: build
	echo "Start Build"

#Common Options
KOKKOS_CUDA_OPTIONS = "enable_lambda"

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/config/nvcc_wrapper
EXE = ${EXE_NAME}.cuda
else
CXX = g++
EXE = ${EXE_NAME}.host
endif

CXXFLAGS = -O3 -g

LINK = ${CXX}
LINKFLAGS = -O3

DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean 
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<
