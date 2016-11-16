#/bin/bash

TEST_DIR="test_tmp"

mkdir $TEST_DIR

$ERROR_SUM

for PAGE_NUMBER in 1 2 3 4 5 6
do
  echo "page "$PAGE_NUMBER

  ./ocr "dataset/page "$PAGE_NUMBER" small.png" > $TEST_DIR"/"$PAGE_NUMBER".txt"

  python "compare files.py" $TEST_DIR"/"$PAGE_NUMBER".txt" "dataset/page "$PAGE_NUMBER".txt"

  ERROR=$?
  echo "error: "$ERROR
  ERROR_SUM=$(($ERROR_SUM+$ERROR))
done

echo "--------"
echo "total error: "$ERROR_SUM

rm -r $TEST_DIR
