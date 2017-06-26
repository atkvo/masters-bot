CC=g++
SRC=VisionSettings.cpp VisionClassifier.cpp vision.cpp
OUTPUT=vision.out
BUILD=build

all: build
	./$(BUILD)/$(OUTPUT)

build: $(BUILD)/$(OUTPUT)
	echo "Built"

$(BUILD)/$(OUTPUT):
	mkdir -p $(BUILD)
	$(CC) $(SRC) -o $(BUILD)/$(OUTPUT) `pkg-config --cflags --libs opencv`

clean:
	rm $(BUILD)/$(OUTPUT)

