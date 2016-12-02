#bin/bash

for C in a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9 period questionmark dash comma; do
  SYMBOL_FOLDER="./chars/no noise/$C/"

  ls -A -w 1 "$SYMBOL_FOLDER" > tmp_files.txt
  cat tmp_files.txt | grep -e "^[0-9]*.png" -e "average.png" | sort -n > tmp_files_ok.txt

  LAST_NUMBER=$(tail -n 1 tmp_files_ok.txt | cut -d "." -f1)

  while read FILENAME; do
    if ! grep -Fxq "$FILENAME" tmp_files_ok.txt; then
      LAST_NUMBER=$((LAST_NUMBER+1))
      mv "$SYMBOL_FOLDER$FILENAME" "$SYMBOL_FOLDER$LAST_NUMBER.png"
    fi
  done < tmp_files.txt

  rm tmp_files.txt tmp_files_ok.txt
done
