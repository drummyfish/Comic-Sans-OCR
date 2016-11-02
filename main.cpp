#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <sstream>

#define DEBUG 1

using namespace std;
using namespace cv;


/*
  Checks if given row (or column) is contains only blank space or not depending on given
  white/black ratio threshold.
*/

bool is_nonblank(Mat input_matrix, bool rows, int row_col_index, double threshold = 0.5)
  {
    double avg;
    avg = mean(rows ? input_matrix.row(row_col_index) : input_matrix.col(row_col_index))[0] / 255.0;
    return avg <= threshold;
  }

int main(int argc, char *argv[])
  {
    if (argc < 2)
      {
        cout << "POV Comic Sans OCR, usage:" << endl;
        cout << "ocr <image filename>" << endl;
        return 0;
      }

    Mat input_image;
    input_image = imread(argv[1],CV_LOAD_IMAGE_COLOR);

    if(!input_image.data)
      {
        cout <<  "Could not open the image." << endl;
        return -1;
      }

    Mat grayscale_thresholded_image;
 
    cvtColor(input_image,grayscale_thresholded_image,CV_BGR2GRAY,1);

adaptiveThreshold(grayscale_thresholded_image,grayscale_thresholded_image,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,5,15);
    
//    threshold(grayscale_thresholded_image,grayscale_thresholded_image,mean(grayscale_thresholded_image)[0],255,THRESH_BINARY);
    imwrite("debug_threshold.png",grayscale_thresholded_image);

    {
      Mat highlighted_lines;
      cvtColor(grayscale_thresholded_image,highlighted_lines,CV_GRAY2RGB,3);

      Mat highlight = Mat(1,highlighted_lines.cols,CV_8UC3,Scalar(100,100,0));

      for (int i = 0; i < grayscale_thresholded_image.rows; i++)
        {
          if (is_nonblank(grayscale_thresholded_image,true,i,0.99))
            {
              highlighted_lines.row(i) -= highlight;
              //highlighted_lines.at<char>(i,0) = 0;
            }
        }

      imwrite("debug_lines.png",highlighted_lines);
    }

    return 0;
  }
