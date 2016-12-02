#/bin/bash

TOTAL_TIME=0

function measure_start
  {
    START_TIME=$(date +%s.%N)
  }

function measure_end
  {
    END_TIME=$(date +%s.%N)
    TIME_DIFF=$(echo "$END_TIME - $START_TIME" | bc)
    echo "time: " $TIME_DIFF
    TOTAL_TIME=$(echo "$TOTAL_TIME + $TIME_DIFF" | bc)
  }

function reset_time
  {
    echo "total time: " $TOTAL_TIME
    TOTAL_TIME=0
  }

TEST_DIR="test_tmp"

rm -r $TEST_DIR
mkdir $TEST_DIR




if true; then

ERROR_SUM=0

echo "=== POV OCR, simple classifier, small pages ==="

for PAGE_NUMBER in 1 2 3 4 5 6
do
  echo "page "$PAGE_NUMBER

  measure_start
  ./ocr "dataset/page "$PAGE_NUMBER" small.png" > $TEST_DIR"/"$PAGE_NUMBER".txt"
  measure_end

  ERROR=$(python "compare files.py" $TEST_DIR"/"$PAGE_NUMBER".txt" "dataset/page "$PAGE_NUMBER".txt")

  echo "error: "$ERROR
  ERROR_SUM=$(($ERROR_SUM+$ERROR))
done

echo "--------"
reset_time
echo "total error: "$ERROR_SUM

fi








if true; then

ERROR_SUM=0

echo "=========== tesseract, small pages ============"

for PAGE_NUMBER in 1 2 3 4 5 6
do
  echo "page "$PAGE_NUMBER

  measure_start
  tesseract "dataset/page "$PAGE_NUMBER" small.png" $TEST_DIR"/"$PAGE_NUMBER"tess_a"
  measure_end

  cat $TEST_DIR"/"$PAGE_NUMBER"tess_a.txt" | sed '/^\s*$/d' > $TEST_DIR"/"$PAGE_NUMBER"tess.txt" # remove empty lines

  ERROR=$(python "compare files.py" $TEST_DIR"/"$PAGE_NUMBER"tess.txt" "dataset/page "$PAGE_NUMBER".txt")

  echo "error: "$ERROR
  ERROR_SUM=$(($ERROR_SUM+$ERROR))
done

echo "--------"
reset_time
echo "total error: "$ERROR_SUM

fi

#rm -r $TEST_DIR
