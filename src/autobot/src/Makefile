CC=g++
SRC=VisionClassifier.cpp vision.cpp
OUTPUT=vision.out
BUILD=build

all: $(BUILD)/$(OUTPUT)
	./$(BUILD)/$(OUTPUT)

$(BUILD)/$(OUTPUT):
	mkdir -p $(BUILD)
	$(CC) $(SRC) -o $(BUILD)/$(OUTPUT) `pkg-config --cflags --libs opencv`

clean:
	rm $(BUILD)/$(OUTPUT)

