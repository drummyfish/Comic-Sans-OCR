#/bin/bash

function run_test
  # params:
  # $1 0 (POV OCR) or 1 (tesseract)
  # $2 filename (without path)
  # $3 reference filename (without path)
  # $4 classifier number for POV OCR
  {
    pathname="dataset/"
    filename="$pathname$2"
    reference_filename="$pathname$3"
    output_filename="tmp.txt"

    rm knn_features.xml
    rm neural_weights.xml

    if [ $1 -eq 0 ]; then
      echo "testing POV OCR, $2, classifier = $4"
      to_run="./ocr \"$filename\" $4 > $output_filename"
    else
      echo "testing tesseract"
      to_run="tesseract \"$filename\" stdout | sed '/^\s*$/d' > $output_filename"
    fi

    start_time=$(date +%s.%N)
    eval ""$to_run""
    end_time=$(date +%s.%N)
    time_diff=$(echo "$end_time - $start_time" | bc)
    echo "  time: $time_diff s"

    if [ $# -ge 4 ] && [ $4 -eq 0 ]; then    # only segmentation, replace everything with "?"
      echo "  (testing segmentation only)"
      cat "$reference_filename" | sed "s/[^ ]/?/g" > tmp2.txt
      reference_filename="tmp2.txt"
    fi

    error=$(python "compare files.py" "$output_filename" "$reference_filename")
    echo "  error: $error"
  }

run_test 1 "page 1 small.png" "page 1.txt"
run_test 0 "page 1 small.png" "page 1.txt" 0
run_test 0 "page 1 small.png" "page 1.txt" 1
run_test 0 "page 1 small.png" "page 1.txt" 2
run_test 0 "page 1 small.png" "page 1.txt" 3
run_test 0 "page 1 small.png" "page 1.txt" 4
run_test 0 "page 1 small.png" "page 1.txt" 5

run_test 1 "page 2 small.png" "page 2.txt"
run_test 0 "page 2 small.png" "page 2.txt" 0
run_test 0 "page 2 small.png" "page 2.txt" 1
run_test 0 "page 2 small.png" "page 2.txt" 2
run_test 0 "page 2 small.png" "page 2.txt" 3
run_test 0 "page 2 small.png" "page 2.txt" 4
run_test 0 "page 2 small.png" "page 2.txt" 5

run_test 1 "page 3 small.png" "page 3.txt"
run_test 0 "page 3 small.png" "page 3.txt" 0
run_test 0 "page 3 small.png" "page 3.txt" 1
run_test 0 "page 3 small.png" "page 3.txt" 2
run_test 0 "page 3 small.png" "page 3.txt" 3
run_test 0 "page 3 small.png" "page 3.txt" 4
run_test 0 "page 3 small.png" "page 3.txt" 5

run_test 1 "page 4 small.png" "page 4.txt"
run_test 0 "page 4 small.png" "page 4.txt" 0
run_test 0 "page 4 small.png" "page 4.txt" 1
run_test 0 "page 4 small.png" "page 4.txt" 2
run_test 0 "page 4 small.png" "page 4.txt" 3
run_test 0 "page 4 small.png" "page 4.txt" 4
run_test 0 "page 4 small.png" "page 4.txt" 5

run_test 1 "page 5 small.png" "page 5.txt"
run_test 0 "page 5 small.png" "page 5.txt" 0
run_test 0 "page 5 small.png" "page 5.txt" 1
run_test 0 "page 5 small.png" "page 5.txt" 2
run_test 0 "page 5 small.png" "page 5.txt" 3
run_test 0 "page 5 small.png" "page 5.txt" 4
run_test 0 "page 5 small.png" "page 5.txt" 5

run_test 1 "page 6 small.png" "page 6.txt"
run_test 0 "page 6 small.png" "page 6.txt" 0
run_test 0 "page 6 small.png" "page 6.txt" 1
run_test 0 "page 6 small.png" "page 6.txt" 2
run_test 0 "page 6 small.png" "page 6.txt" 3
run_test 0 "page 6 small.png" "page 6.txt" 4
run_test 0 "page 6 small.png" "page 6.txt" 5

run_test 1 "page 7 small.png" "page 7.txt"
run_test 0 "page 7 small.png" "page 7.txt" 0
run_test 0 "page 7 small.png" "page 7.txt" 1
run_test 0 "page 7 small.png" "page 7.txt" 2
run_test 0 "page 7 small.png" "page 7.txt" 3
run_test 0 "page 7 small.png" "page 7.txt" 4
run_test 0 "page 7 small.png" "page 7.txt" 5

# other:

run_test 1 "page 1.png" "page 1.txt"
run_test 0 "page 1.png" "page 1.txt"

run_test 1 "page 2.png" "page 2.txt"
run_test 0 "page 2.png" "page 2.txt"

run_test 1 "page 1 small noise.png" "page 1.txt"
run_test 0 "page 1 small noise.png" "page 1.txt"

run_test 1 "page 1 noise.png" "page 1.txt"
run_test 0 "page 1 noise.png" "page 1.txt"
