HEADERS = types.h utils.h gpu_single.h
OBJECTS = main.o

CC = nvcc

default: main

%.o: %.cu $(HEADERS)
	$(CC) -c $< -o $@

program: $(OBJECTS)
	$(CC)  $(OBJECTS) -o $@

clean:
	-rm -f $(OBJECTS)
	-rm -f main