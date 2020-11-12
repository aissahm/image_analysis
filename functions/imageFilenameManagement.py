#given a text file containing the name of training images return pathfile of respective masks
def returnMaskImagesPathArray(trainTextPath):
  mask_images_path_array = []
  train_images_array = []
  with open(trainTextPath, 'r') as file:
      data = file.read().replace('\n', ';')
      train_images_array = data.split(";")
  for train_image_str in train_images_array:
    mask_image_str = train_image_str.replace("./plates/orig", "/content/plates/mask")
    if len(mask_image_str) > 0:
      mask_images_path_array.append(mask_image_str)

  return mask_images_path_array

#given a text file containing the name of training images return pathfile of respective images
def returnImagesPathArray(trainTextPath):
  images_path_array = []
  train_images_array = []
  with open(trainTextPath, 'r') as file:
      data = file.read().replace('\n', ';')
      train_images_array = data.split(";")
  for train_image_str in train_images_array:
    image_str = train_image_str.replace("./plates", "/content/plates")
    if len(image_str) > 0:
      images_path_array.append(image_str)
  return images_path_array

 #returns mask for correspoinding image
def returnMaskPathFromImagePath(imagePath):
  return imagePath.replace("orig", "mask")

#returns the ID of the original image
def returnOriginalimagename(imagename):
  originalImagename = imagename.replace(".png", "")
  originalImagename = originalImagename.replace(" ", "")
  return originalImagename

#returns information about image from filename
 def returnSubimageDimensionsCenterCoordinates(imagename):
  imagename = imagename.replace(".png", "")
  elementsArray = imagename.split("_")
  imageID = elementsArray[1]
  width = int(elementsArray[2])
  height = int(elementsArray[3])
  xcenter = int(elementsArray[4])
  ycenter = int(elementsArray[5])
  imageclass = int(elementsArray[6])
  return [imageID, width, height, xcenter, ycenter, imageclass]
