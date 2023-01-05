OBJS		= rapid.o build.o collide.o overlap.o 

OS_NAME := $(shell uname -s | tr A-Z a-z)

CFLAGS		= -Wall -fPIC -O2 -I. 
CXXFLAGS	= $(CFLAGS)
CPPFLAGS	= -pedantic

ifeq ($(OS_NAME),linux)
	TARGET = libRAPID.so
	FLAG = -soname
else 
	TARGET = libRAPID.dylib
	FLAG = -install_name
endif

all: $(OBJS)
	$(CXX) $(CFLAGS) -c rapidAPI.cc
	$(CXX) -shared -Wl,$(FLAG),$(TARGET) -o $(TARGET) $(OBJS) rapidAPI.o  
	-cp $(TARGET) ..

$(OBJS): %.o: %.c
	$(CXX) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	-rm *.so
	-rm *.o
