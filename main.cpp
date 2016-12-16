/**
  Comic Sans OCR - FIT BUT school project
  2016
  Miloslav Číž
  Daniel Žůrek
 */

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <opencv2/features2d/features2d.hpp>
#include <opencv/ml.h>
#include <cstdlib>
#include <stdio.h>
#include <dirent.h>
#include <string.h>

#define DEBUG true
#define DEBUG_CUTOUT_RESIZE 32
#define DEBUG_CUTOUTS_MAX 50
#define DEBUG_CUTOUTS_START 100

#define DATASET_LOCATION "dataset/chars/no noise/"

#define MIN_CHAR_WIDTH_TO_IMAGE_WIDTH_RATIO 0.0025
#define MIN_CHAR_HEIGHT_TO_IMAGE_WIDTH_RATIO 0.005
#define SPACE_TO_LINE_HEIGHT_RATIO 0.3             // how many pixels is considered a space
#define LINE_BRIGHTNESS_THRESHOLD 0.99
#define COLUMN_BRIGHTNESS_THRESHOLD 0.97
#define MAX_SEGMENT_WIDTH_TO_HEIGHT_RATIO 1.1      // if segment's w/h ratio is bigger, it will be split

#define DATASET_LOCATION "dataset/chars/no noise/"
#define COMMA 44
#define DASH 45
#define PERIOD 46
#define QUESTIONMARK 63
#define KNN_FILENAME "knn_features.xml"

using namespace std;
using namespace cv;

typedef struct
  {
    unsigned int start;
    unsigned int length;
  } segment_1d;

#define CLASSIFIER_NONE 0
#define CLASSIFIER_SIMPLE 1
#define CLASSIFIER_KNN 2

int classifier_to_use;      ///< says which classifier will be used, see constants starting with CLASSIFIER_

unsigned char POSSIBLE_CHARACTERS[] =
  {
    'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    '0','1','2','3','4','5','6','7','8','9',
    '?',/*'-',*/',','.'
  };

Mat average_character_images[sizeof(POSSIBLE_CHARACTERS)];
double average_w_to_h_ratios[sizeof(POSSIBLE_CHARACTERS)];

class OcrKnn    ///< KNN for OCR
  {
    public:
      OcrKnn();
      void load_data();                          ///< loads the images and prepares the classes
      void prepare_knn();                        ///< gets HOG feature for each image and adds it to the train data
      float classify(Mat input_image);
      void train();
      void test();

      bool save_to_file();
      bool load_from_file();

      Mat image_preprocess(Mat input_image);
      Mat image_deskew(Mat input_image);         ///< corrects the image skew
      Mat get_hog_descriptor(Mat input_image);   ///< gets a HOG descriptor of the image

    private:
      int k;
      Mat train_data, train_classes;
      KNearest knn;
      vector<string> images;
      vector<int> classes;
  };

OcrKnn::OcrKnn()
  {
  }

Mat OcrKnn::image_deskew(Mat input_image)
  {
    Mat thr;
    threshold(input_image, thr, 200, 255, THRESH_BINARY_INV);

    vector<Point> points;
    Mat_<uchar>::iterator it = thr.begin<uchar>();
    Mat_<uchar>::iterator end = thr.end<uchar>();

    for (; it != end; ++it)
      if (*it) points.push_back(it.pos());
      	
    RotatedRect box = minAreaRect(Mat(points));
    Mat rot_mat = getRotationMatrix2D(box.center, box.angle, 1);

    Mat rotated;
    warpAffine(input_image, rotated, rot_mat, input_image.size(), INTER_CUBIC);
	
    return rotated;
  }

Mat OcrKnn::get_hog_descriptor(Mat preprocesed_image)
  {
    Mat Hogfeat;
    HOGDescriptor hogDescriptor(Size(32, 16), Size(8, 8), Size(4, 4), Size(4, 4), 9);

    vector<float> descriptorsValues;
    vector<Point> locations;

    hogDescriptor.compute(preprocesed_image, descriptorsValues, Size(32, 32), Size(0, 0), locations);
    Hogfeat.create(descriptorsValues.size(), 1, CV_32FC1);

    for (unsigned int j = 0; j < descriptorsValues.size(); j++)
      Hogfeat.at<float>(j, 0) = descriptorsValues.at(j);
      
    return Hogfeat.reshape(1, 1); 
  }

Mat OcrKnn::image_preprocess(Mat input_image)
  {
    Mat preprocessed_image;

    // resize
    Mat resized;
    resize(input_image, resized, Size(64, 48));

    // correct skew 
    Mat deskewed = image_deskew(resized);

    // to grayscale
    Mat gray_image;
    cvtColor(deskewed, gray_image, CV_BGR2GRAY);

    // Gaussian blur
    Mat gaussian_image;
    GaussianBlur(gray_image, gaussian_image, Size(3,3), 0, 0);

    //Mat adap_thres_image;
    //adaptiveThreshold(gaussian_image, adap_thres_image, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 11, 2);

    preprocessed_image = gaussian_image;
    return preprocessed_image;
  }

void OcrKnn::load_data()
  {
    DIR *dp, *fp;
    struct dirent *dirp, *firp;
    char dir_path[255] = "";
    char file_path[255] = "";

    // load the directory

    dp = opendir(DATASET_LOCATION);

    while ((dirp = readdir(dp)) != NULL)
      {
        if (!strcmp(dirp->d_name, ".") || !strcmp(dirp->d_name, "..")) continue;

        strcat(dir_path, DATASET_LOCATION);
        strcat(dir_path, dirp->d_name);
        strcat(dir_path, "/");

        // read the image from the directory

        fp = opendir(dir_path);

        while ((firp = readdir(fp)) != NULL)
          {
            if (!strcmp(firp->d_name, ".") || !strcmp(firp->d_name, "..")) continue;

            // get the class, it is represented by the ASCII value of the character
            if (strlen(dirp->d_name) == 1)
              {
                classes.push_back((int)dirp->d_name[0]);
              }
            else
              {
                if (strcmp(dirp->d_name, "comma")) classes.push_back(COMMA);
                else if (strcmp(dirp->d_name, "dash")) classes.push_back(DASH);
                else if (strcmp(dirp->d_name, "period")) classes.push_back(PERIOD);
                else if (strcmp(dirp->d_name, "questionmark")) classes.push_back(QUESTIONMARK);
              }

            strcat(file_path, dir_path);
            strcat(file_path, firp->d_name);
            string str(file_path);
            images.push_back(str);

            file_path[0] = '\0';
	  }
        
        closedir(fp);
        dir_path[0] = '\0';
      }

    closedir(dp);
  }

void OcrKnn::prepare_knn()
  {
    Mat src_image, preprocessed_image, hogDescriptor;

    // Naplneni trid(labels) pro obrazky
    train_classes = Mat(classes).reshape(0, classes.size());

    for (unsigned int i = 0; i < images.size(); i++)
      {
        src_image = imread(images.at(i), CV_LOAD_IMAGE_COLOR);
        preprocessed_image = image_preprocess(src_image);
        hogDescriptor = get_hog_descriptor(preprocessed_image);
        train_data.push_back(hogDescriptor);
      }
  }

void OcrKnn::train()
  {
    knn.train(train_data, train_classes);
    this->k = knn.get_max_k() / 2;
  }

float OcrKnn::classify(Mat input_image)
  {
    Mat inputImageDescriptor, hogDescriptor;
    Mat preprocessed_image;
    preprocessed_image = image_preprocess(input_image);
    hogDescriptor = get_hog_descriptor(preprocessed_image);
    inputImageDescriptor.push_back(hogDescriptor);
    return knn.find_nearest(inputImageDescriptor,this->k);;
  }

bool OcrKnn::save_to_file()
  {
    FileStorage file_storage(KNN_FILENAME,FileStorage::WRITE);

    if (!file_storage.isOpened())
      return false;

    file_storage << "hog_features" << this->train_data;
    file_storage << "train_classes" << this->train_classes;
    file_storage.release();
    return true;
  }

bool OcrKnn::load_from_file()
  {
    FileStorage file_storage(KNN_FILENAME,FileStorage::READ);

    if (!file_storage.isOpened())
      return false;

    file_storage["hog_features"] >> this->train_data;
    file_storage["train_classes"] >> this->train_classes;
    return true;
  }

//-----------------------------------------------------------

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

    int state = 0;  // looking for segment start

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
            if (state == 1)  // looking for segment end
              {
                helper_segment.length = i - helper_segment.start;

                if (helper_segment.length >= minimum_segment_length)  // segment too small?
                  {
                    if (!vertical && (helper_segment.length / float(input_matrix.rows)) > MAX_SEGMENT_WIDTH_TO_HEIGHT_RATIO)  // segment too fat?
                      {
                        // split the segment in two
                        segment_1d helper_segment2;

                        // we split the segment in half, might be better to do on minimum
                        helper_segment.length /= 2;

                        helper_segment2.start = helper_segment.start + helper_segment.length;
                        helper_segment2.length = helper_segment.length;

                        result.push_back(helper_segment);
                        result.push_back(helper_segment2);                      
                      }
                    else
                      {
                        result.push_back(helper_segment);   // add the segment as is
                      }
                  }

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
        cout << "ocr <image filename> [classifier]" << endl << endl;
        cout << "classifier can be:" << endl;
        cout << "  0 - none" << endl;
        cout << "  1 - simple (average image comparison)" << endl;
        cout << "  2 - KNN (default)" << endl;
        cout << endl;
        cout << "To retrain the classifier, delete the .xml files." << endl;
        return 0;
      }

    if (argc >= 3)
      {
        switch (argv[2][0])
          {
            case '0': classifier_to_use = CLASSIFIER_NONE; break;
            case '1': classifier_to_use = CLASSIFIER_SIMPLE; break;
            case '2': classifier_to_use = CLASSIFIER_KNN; break;
            default: classifier_to_use = CLASSIFIER_KNN; break;
          }
      }
    else
      classifier_to_use = CLASSIFIER_KNN;

    OcrKnn ocr_knn;

    switch (classifier_to_use)  // init given classifier
      {
        case CLASSIFIER_SIMPLE:
          for (unsigned int i = 0; i < sizeof(POSSIBLE_CHARACTERS); i++)
            {
              string filename = DATASET_LOCATION + character_to_filesystem_name(POSSIBLE_CHARACTERS[i]) + "/average.png";
              average_character_images[i] = imread(filename,CV_LOAD_IMAGE_COLOR);
              cvtColor(average_character_images[i],average_character_images[i],CV_RGB2GRAY);
              average_w_to_h_ratios[i] = average_character_images[i].cols / ((double) average_character_images[i].rows);
            }
          break;
 
        case CLASSIFIER_KNN:
          if (!ocr_knn.load_from_file())
            { // no file => retrain
              ocr_knn.load_data();
              ocr_knn.prepare_knn();
              ocr_knn.save_to_file();
            }

          ocr_knn.train();
          break;

        default:
          break;
      } 

    Mat input_image;
    input_image = imread(argv[1],CV_LOAD_IMAGE_COLOR);

    Mat debug_cutouts(DEBUG_CUTOUT_RESIZE,DEBUG_CUTOUT_RESIZE,input_image.type());
    int cutout_append_counter = 0;

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

    // segment lines:
    vector<segment_1d> lines = segmentation_1d(grayscale_thresholded_image,true,LINE_BRIGHTNESS_THRESHOLD,min_char_height);

    {
      Mat highlighted_lines;
      cvtColor(grayscale_thresholded_image,highlighted_lines,CV_GRAY2RGB,3);

      unsigned int character_number = 0;

      for (unsigned int i = 0; i < lines.size(); i++)  // for each segmented line
        {
          Mat line_image(highlighted_lines,Rect(0,lines[i].start,highlighted_lines.cols,lines[i].length));

          vector<segment_1d> columns = segmentation_1d(line_image,false,COLUMN_BRIGHTNESS_THRESHOLD,min_char_width);

          unsigned int space_pixels = (int) (SPACE_TO_LINE_HEIGHT_RATIO * lines[i].length);       // how many pixels is considered a space
          unsigned int previous_character_stop = 0;                                               // for detecting spaces
          
          for (unsigned int j = 0; j < columns.size(); j++)  // for each column in the line
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

                  /** saves characters as images, for making the dataset
                  if (character_number > 200)
                    imwrite("chars/img_" + std::to_string(character_number) + ".png",character_cutout);
                  */

                  char recognised_character;

                  switch (classifier_to_use)
                    {
                      case CLASSIFIER_NONE:
                        recognised_character = '?';
                        break;

                      case CLASSIFIER_SIMPLE:
                        recognised_character = classify_simple(character_cutout);
                        break;

                      case CLASSIFIER_KNN:
                        recognised_character = ocr_knn.classify(character_cutout);
                        break;

                      default:
                        break;
                    }

                  if (DEBUG && character_number >= DEBUG_CUTOUTS_START && cutout_append_counter < DEBUG_CUTOUTS_MAX)
                    {
                      int new_width = int((float(DEBUG_CUTOUT_RESIZE) / character_cutout.rows) * character_cutout.cols);
                      new_width = new_width == 0 ? 1 : new_width;

                      resize(character_cutout,character_cutout,Size(new_width,DEBUG_CUTOUT_RESIZE));
 
                      if (cutout_append_counter < 1)
                        {
                          hconcat(&character_cutout,1,debug_cutouts);
                        }
                      else
                        {
                          Mat separator(DEBUG_CUTOUT_RESIZE,1,input_image.type());
                          separator = cv::Scalar(0,0,255);
                          Mat mat_array[] = {debug_cutouts,separator,character_cutout};
                          hconcat(mat_array,3,debug_cutouts);
                        }

                      cutout_append_counter++;
                    }

                  cout << recognised_character;

                  char_image -= Scalar(100,100,0);  // highlight the character
                  character_number++;
                }
            }

          cout << endl;

          line_image -= Scalar(0,50,0);         // highlight the line
        }

      if (DEBUG)
        {
          imwrite("debug_segments.png",highlighted_lines);
          imwrite("debug_cutouts.png",debug_cutouts);
        }
    }

    return 0;
  }
