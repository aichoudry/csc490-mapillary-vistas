  # CSC490: Mapillary Vistas Dataset
  ### Abdurrahman Choudry, Ryan D’Rosario, Hanchao Ge, Matthew Ho Kin

#### Problem
  The problem we addressed is semantic segmentation for street scenes, namely, pixel by pixel segmentation of a street image into a set of classes. 

#### Mapillary Vistas Dataset
Dataset here: https://www.mapillary.com/dataset/vistas
The dataset we’re using is the Mapillary Vistas v1.2 Dataset. This dataset contains 25,000 images along with 66 object categories. It is pixel-accurate and cover six continents which makes it applicable to a good portion of the world. In addition to this, the images contain enough variety to be applicable in a plethora of situations, such as weather conditions, camera angles, times of day etc. The devices used to create the pictures were also different as were the photographers who took the pictures and who have different levels of experience.

##### Input Example:
![Input Example](https://github.com/aichoudry/csc490-mapillary-vistas/blob/main/sample_data/North%20America/aSqVUgt36gddhmJdI1lXNA.jpg?raw=true)
##### Output Example:
![Output Example](https://github.com/aichoudry/csc490-mapillary-vistas/blob/main/sample_data/North%20America/aSqVUgt36gddhmJdI1lXNA.png?raw=true)

#### Implementation Decisions
##### Data Augmentation Approaches
There were some problems we identified with the dataset, the main one being that the images were very large (at least 1000x1000, upwards of 3000x4000). Some pre-processing steps needed so that the training would finish in a reasonable amount of time and use a reasonable amount of memory. There were two main approaches we took:
###### Resize
Resizing consisted of simple resizing the image to a fixed size. This was a common approach for some of the other models we've seen that used the dataset. However it came with problems such as, introducing artifacting in the image, removing detail of objects, and removing classes that only appeared in a small portion of the image. We did not use this approach going forward after discovering these issues.
###### Fixed-Crop
Fixed-Crop consisted of cropping the image at a fixed location in the image at a specific size. There were 4 crops we tested, LEFT | RIGHT | CENTRE | RANDOM. Each had there own advantages and disadvantages. LEFT & RIGHT crop performed quite poorly as the diversity of classes in those locations for most images is limited. It is also not exposed to some of the classes that appear often as much, such as SKY. CENTRE & RANDOM were much better in this regard, as the CENTRE of the image tends to get exposed to many more classes and a RANDOM crop won't be bias towards some classes or others as the location it crops is random. 

##### UNet
The UNet acchitecture, as the name suggest, forms a CNN in the shape of a U. The model is built with 3 parts in mind. The first part is the encoder, the second is the bottle neck and the third is the decoder. There are skip connections between the encoder and decoder which means that some outputs of the encoder are passed in as inputs to the decoder. The encoder takes the input image and uses convolutions to extract the features within the image. It uses 2 convolutions followed by 2 ReLUs respectively before doing a max pool to reduce the size of the image while also storing the output of the current layer for each layer of the network. The bottleneck further reduces the parameters to avoid a computionally expensive action with another convolution. The decoder then takes the features and skip connections and uses transpose convolutions to output the classification done on the images. The skip connections are done between the encoder and decoder when the shapes of the output of the encoder and input of the decoder are the same. This forms a U shape where the encoder reduces the shape going down the U, the bottleneck is the plateau of the U and the decoder restores the shape going up the U with skip connections that are effectively a straight horizontal line through the U. This architecture was developed for medical imaging and is useful for image segmentation and it has  been shown to have good results with the medical classification. (edited)
5 December 2023

##### ResNet + UNet
The ResnetUNet branch was developed using Fastai, which is a library built on Pyvision with the intention of speeding up model assembly. The unet_learner class was specifically helpful in constructing the Resnet encoder with UNet decoder. Note that Fastai lacks certain features that you would normally find it the base Pytorch library, such as checkpoint saves, metric plotting.
Checkpoints are implemented as a model callback during training, and metric plotting is provided by Ignacio Oguiza at 'https://forums.fast.ai/t/plotting-metrics-after-learning/69937'.

Torchvision is another large component of the provided scripts as the library provides many useful utilities for manipulating images and tensors.

See 'www.fast.ai' for more details. 

#### Results
See the folders prefixed with _model_ _unet_ to see results of specific models.

#### Contribution
**Abdurrahman Choudry:** Built the UNet model and created the main scripts for transforming, training and testing it. My focus was on researching and testing different loss functions (weighted) and data augmentation methods/image transformations we could do to minmize the time to train and memory usage while also maintain or increase accuracy  
**Matthew Ho Kin:** First through the research of the UNet model and then all the development and training of the Resnet-UNet model using the Fastai library.  
**Ryan D’Rosario:** Worked on getting the values for the hyperparameters to optimize the model.  
**Hanchao Ge:** Displaying output images with palette, training and validating Resnet-UNet model
