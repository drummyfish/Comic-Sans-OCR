#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <sstream>
#include <vector>

#define DEBUG true

#define MIN_CHAR_WIDTH_TO_IMAGE_WIDTH_RATIO 0.0025
#define MIN_CHAR_HEIGHT_TO_IMAGE_WIDTH_RATIO 0.005
#define SPACE_TO_LINE_HEIGHT_RATIO 0.3         // how many pixels is considered a space
#define LINE_BRIGHTNESS_THRESHOLD 0.99
#define COLUMN_BRIGHTNESS_THRESHOLD 0.97

using namespace std;
using namespace cv;

#define DATASET_LOCATION "dataset/chars/no noise/"

typedef struct
  {
    unsigned int start;
    unsigned int length;
  } segment_1d;

int classifier_to_use;      // says which classifier will be used: 0 - simple classifier, TODO

unsigned char POSSIBLE_CHARACTERS[] =
  {
    'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    '0','1','2','3','4','5','6','7','8','9',
    '?',/*'-',*/',','.'
  };

Mat average_character_images[sizeof(POSSIBLE_CHARACTERS)];
double average_w_to_h_ratios[sizeof(POSSIBLE_CHARACTERS)];

/*
  Converts an ASCII character to a name used for files in the dataset.
  */

string character_to_filesystem_name(unsigned char input_character)
  {
    switch (input_character)
      {
        case '?': return "questionmark"; break;
        case '-': return "dash"; break;
        case '.': return "period"; break;
        case ',': return "comma"; break;
        default: return string(1,input_character); break;
      }
  }

/*
  Performs a simple classification of given sample by comparing it to average images of
  characters. Returns the ASCII value of the character.
  */

unsigned char classify_simple(Mat input_image)
  {
    double minimum_error = 9999999999999.0;
    unsigned int minimum_error_index = 0;

    double w_to_h_ratio = input_image.cols / ((double) input_image.rows);

    Mat image_copy = input_image.clone();
    cvtColor(image_copy,image_copy,CV_RGB2GRAY);
   // adaptiveThreshold(image_copy,image_copy,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,5,15);

    for (unsigned int i = 0; i < sizeof(POSSIBLE_CHARACTERS); i++)
      {
        Mat difference;
        Mat resized_average_image;

        resize(average_character_images[i],resized_average_image,Size(input_image.cols,input_image.rows));

        absdiff(image_copy,resized_average_image,difference);
        double error = mean(difference)[0] / 255.0;

        double side_ratio_error = abs(w_to_h_ratio - average_w_to_h_ratios[i]) / 2.0;

        error += side_ratio_error;

     //   cout << error << endl;

        if (error < minimum_error)
          {
            minimum_error = error;
            minimum_error_index = i;
          }
      } 

    return POSSIBLE_CHARACTERS[minimum_error_index];
  }

/*
  Checks if given row (or column) is contains only blank space or not depending on given
  white/black ratio threshold. */

bool is_nonblank(Mat input_matrix, bool rows, int row_col_index, double threshold = 0.5)
  {
    double avg;

    avg = mean(rows ? input_matrix.row(row_col_index) : input_matrix.col(row_col_index))[0] / 255.0;
    return avg <= threshold;
  }

/*
  Performs a 1D segmentation on either image rows or columns. */

vector<segment_1d> segmentation_1d(Mat input_matrix, bool vertical, double threshold, unsigned int minimum_segment_length)
  {
    vector<segment_1d> result;
    segment_1d helper_segment;

    int state = 0;    // looking for segment start

    for (int i = 0; i < (vertical ? input_matrix.rows : input_matrix.cols); i++)
      {
        if (is_nonblank(input_matrix,vertical,i,threshold))
          {
            if (state == 0)
             {
               helper_segment.start = i;
               state = 1;
             }
          }
        else
          {
            if (state == 1)
              {
                helper_segment.length = i - helper_segment.start;

                if (helper_segment.length >= minimum_segment_length)
                  result.push_back(helper_segment);

                state = 0;
              }
          }
      }

    return result;
  }

/*
  Corrects a detected character region. */

Rect correct_character_cutout(Mat input_image, Rect image_cutout, double brightness_threshold)
  {
    Rect result = Rect(image_cutout);

    Mat column_image = Mat(input_image,Rect(image_cutout.x,0,image_cutout.width,input_image.rows));

    for (int i = 0; i < image_cutout.height; i++)          // shrink if possible
      {
        bool go_on = false;

        if (!is_nonblank(column_image,true,result.y,brightness_threshold))
          {
            result.y += 1;
            result.height -= 1;
            go_on = true;
          }

        if (!is_nonblank(column_image,true,result.y + result.height - 1,brightness_threshold))
          {
            result.height -= 1;
            go_on = true;
          }

        if (!go_on)
          break;
      }

    for (int i = 0; i < 20; i++)     // expand up by 20 px if possible
      {
        bool go_on = false;

        if (is_nonblank(column_image,true,result.y - 1,brightness_threshold))
          {
            result.y -= 1;
            result.height += 1;
            go_on = true;
          }

        if (is_nonblank(column_image,true,result.y + result.height,brightness_threshold))
          {
            result.height += 1;
            go_on = true;
          }

        if (!go_on)
          break;
      }

    return result;
  }

void print_segment_1d(segment_1d s)
  {
    cout << "segment: " << s.start << " " << s.length << endl;
  }

int main(int argc, char *argv[])
  {
    if (argc < 2)
      {
        cout << "POV Comic Sans OCR, usage:" << endl;
        cout << "ocr <image filename>" << endl;
        return 0;
      }

    classifier_to_use = 0;   // TODO: read from parameters

    if (classifier_to_use == 0)       // load average images if using a classifier that requires them
      {
        for (unsigned int i = 0; i < sizeof(POSSIBLE_CHARACTERS); i++)
          {
            string filename = DATASET_LOCATION + character_to_filesystem_name(POSSIBLE_CHARACTERS[i]) + "/average.png";
            average_character_images[i] = imread(filename,CV_LOAD_IMAGE_COLOR);
            cvtColor(average_character_images[i],average_character_images[i],CV_RGB2GRAY);
            average_w_to_h_ratios[i] = average_character_images[i].cols / ((double) average_character_images[i].rows);
          //  adaptiveThreshold(average_character_images[i],average_character_images[i],255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,5,15);            

            /*
            namedWindow("Display window",WINDOW_AUTOSIZE);
            imshow("Display window",average_character_images[i]);                
            waitKey(0); */
          }
      }

    Mat input_image;
    input_image = imread(argv[1],CV_LOAD_IMAGE_COLOR);

    unsigned int min_char_width = (int) (MIN_CHAR_WIDTH_TO_IMAGE_WIDTH_RATIO * input_image.cols);
    unsigned int min_char_height = (int) (MIN_CHAR_HEIGHT_TO_IMAGE_WIDTH_RATIO * input_image.cols);

    if(!input_image.data)
      {
        cout <<  "Could not open the image." << endl;
        return -1;
      }

    Mat grayscale_thresholded_image;
 
    cvtColor(input_image,grayscale_thresholded_image,CV_BGR2GRAY,1);

    adaptiveThreshold(grayscale_thresholded_image,grayscale_thresholded_image,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,5,15);
    vector<segment_1d> lines = segmentation_1d(grayscale_thresholded_image,true,LINE_BRIGHTNESS_THRESHOLD,min_char_height);

    {
      Mat highlighted_lines;
      cvtColor(grayscale_thresholded_image,highlighted_lines,CV_GRAY2RGB,3);

      for (unsigned int i = 0; i < lines.size(); i++)
        {
          Mat line_image(highlighted_lines,Rect(0,lines[i].start,highlighted_lines.cols,lines[i].length));

          vector<segment_1d> columns = segmentation_1d(line_image,false,COLUMN_BRIGHTNESS_THRESHOLD,min_char_width);

          unsigned int space_pixels = (int) (SPACE_TO_LINE_HEIGHT_RATIO * lines[i].length);       // how many pixels is considered a space
          unsigned int previous_character_stop = 0;                                               // for detecting spaces
          
          for (unsigned int j = 0; j < columns.size(); j++)
            {
              Rect char_area = correct_character_cutout(highlighted_lines,Rect(columns[j].start,lines[i].start,columns[j].length,lines[i].length),LINE_BRIGHTNESS_THRESHOLD);

              Mat char_image(highlighted_lines,char_area);

              if (char_area.width >= (int) min_char_width && char_area.height >= (int) min_char_height)
                {
                  // character cutout available here

                  if (j != 0 && (char_area.x - previous_character_stop) >= space_pixels)
                    cout << " ";

                  previous_character_stop = char_area.x + char_area.width;

                  Mat character_cutout = input_image(char_area);

                  char recognised_character = classify_simple(character_cutout);

                  cout << recognised_character;

                  char_image -= Scalar(100,100,0);  // highlight the character
                }
            }

          cout << endl;

          line_image -= Scalar(0,50,0);         // highlight the line
        }

      if (DEBUG)
        imwrite("debug_segments.png",highlighted_lines);
    }

    return 0;
  }
