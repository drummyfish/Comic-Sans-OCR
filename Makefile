all:
	g++ main.cpp `pkg-config --libs opencv` -o ocr
run:
	 ./ocr "dataset/page 1 small.png"
run2:
	 ./ocr "dataset/page 2 small.png"
run3:
	 ./ocr "dataset/page 3 small.png"
run4:
	 ./ocr "dataset/page 4 small.png"
run5:
	 ./ocr "dataset/page 5 small.png"
run6:
	 ./ocr "dataset/page 6 small.png"
