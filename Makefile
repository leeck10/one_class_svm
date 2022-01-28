.SUFFIXES: .c .cpp

CC=g++
# linux
#CFLAGS=-I. -O3 -fopenmp
# MinGW
#CFLAGS=-I. -O3 -fno-strict-aliasing -fopenmp
CFLAGS=-I. -O3 -fno-strict-aliasing
#CFLAGS=-I. -g -fno-strict-aliasing 
LD=g++
# linux
#LFLAGS=-O3 -fopenmp
# MinGW
#LFLAGS=-O3 -fno-strict-aliasing -Wl,--large-address-aware -fopenmp
LFLAGS=-O3 -fno-strict-aliasing -Wl,--large-address-aware
#LFLAGS=-g -fno-strict-aliasing -Wl,--large-address-aware
LIBS= 

%.o:	%.cpp
	$(CC) -c -o $@ $(CFLAGS) $<

%.o:	%.c
	gcc -c -o $@ $(CFLAGS) $<

%.o:	%.f
	g77 -c -o $@ $(CFLAGS) $<

# MinGW
target=one_class_svm_tool 
all: $(target)

# one_class_svm
one_class_svm=one_class_svm.o one_class_svm_train.o

one_class_svm_tool: one_class_svm_tool.o $(one_class_svm) \
	cmdline.o getopt.o getopt1.o
	$(LD) -o $@ one_class_svm_tool.o $(one_class_svm) \
	cmdline.o getopt.o getopt1.o \
	$(LIBS) $(LFLAGS)

install:
	cp one_class_svm_tool.exe /usr/local/bin

backup: *.cpp *.hpp *.h *.c
	cp *.[ch]pp backup
	cp *.[chf] backup

clean:
	rm *.o *.a *.lib *.dll *.def *.exp *.so

