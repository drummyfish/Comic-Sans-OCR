from os import listdir
from os.path import isfile, join
from PIL import Image

IMAGE_SIZE = (128,128)
PATH_TO_IMAGES = "chars/no noise"

char_directories = [f for f in listdir(PATH_TO_IMAGES) if not isfile(join(PATH_TO_IMAGES,f))]

for char_directory in char_directories:
  print(char_directory)

  char_files = [f for f in listdir(join(PATH_TO_IMAGES,char_directory)) if isfile(join(PATH_TO_IMAGES,char_directory,f))]

  images = []

  size_sum = [0,0]

  for char_file in char_files:
    print(char_file)
    images.append(Image.open(join(PATH_TO_IMAGES,char_directory,char_file)))
    size_sum[0] += images[-1].size[0]
    size_sum[1] += images[-1].size[1]

  size_sum[0] = int(size_sum[0] / float(len(images)))
  size_sum[1] = int(size_sum[1] / float(len(images)))

  average_image = Image.new("RGB",size_sum,"white")
  average_pixels = average_image.load()

  for i in range(len(images)):
    images[i] = images[i].resize(size_sum)

  for j in range(average_image.size[1]):
    for i in range(average_image.size[0]):
      pixel_sum = [0,0,0]

      for image in images:
        pixels = image.load()
        pixel = pixels[i,j]
        pixel_sum[0] += pixel[0]
        pixel_sum[1] += pixel[1]
        pixel_sum[2] += pixel[2]

      pixel_sum[0] = pixel_sum[0] / len(images)
      pixel_sum[1] = pixel_sum[1] / len(images)
      pixel_sum[2] = pixel_sum[2] / len(images)

      average_pixels[i,j] = tuple(pixel_sum)

  average_image.save(join(PATH_TO_IMAGES,char_directory,"average.png"),"PNG")

