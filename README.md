  # CSC490: Mapillary Vistas Dataset
  ### Abdurrahman Choudry, Ryan D’Rosario, Hanchao Ge, Matthew Ho Kin

#### Problem
  The problem we addressed is semantic segmentation for street scenes, namely, pixel by pixel segmentation of a street image into a set of classes. 
#### The code
The main script is main.py which can be run on the lab machines that have the dataset on it. Update the train.ini file for the parameters of training.

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
Fixed-Crop consisted of cropping the image at a fixed location in the image at a specific size. There were 4 crops we tested, LEFT | RIGHT | CENTRE | RANDOM. Each had there own advantages and disadvantages. LEFT & RIGHT crop performed quite poorly as the diversity of classes in those locations for most images is limited. It is also not exposed to some of the classes that appear often as much, such as SKY. CENTRE & RANDOM were much better in this regard, as the CENTRE of the image tends to get exposed to many more classes and a RANDOM crop won't be bias towards some classes or others as the location it crops is random. This paper helped us with the random crop[5].

##### UNet
The UNet acchitecture, as the name suggest, forms a CNN in the shape of a U. The model is built with 3 parts in mind. The first part is the encoder, the second is the bottle neck and the third is the decoder. There are skip connections between the encoder and decoder which means that some outputs of the encoder are passed in as inputs to the decoder. The encoder takes the input image and uses convolutions to extract the features within the image. It uses 2 convolutions followed by 2 ReLUs respectively before doing a max pool to reduce the size of the image while also storing the output of the current layer for each layer of the network. The bottleneck further reduces the parameters to avoid a computionally expensive action with another convolution. The decoder then takes the features and skip connections and uses transpose convolutions to output the classification done on the images. The skip connections are done between the encoder and decoder when the shapes of the output of the encoder and input of the decoder are the same. This forms a U shape where the encoder reduces the shape going down the U, the bottleneck is the plateau of the U and the decoder restores the shape going up the U with skip connections that are effectively a straight horizontal line through the U. This architecture was developed for medical imaging and is useful for image segmentation and it has  been shown to have good results with the medical classification. 

##### ResNet + UNet
The ResnetUNet branch was developed using Fastai, which is a library built on Pyvision with the intention of speeding up model assembly. The unet_learner class was specifically helpful in constructing the Resnet encoder with UNet decoder. Note that Fastai lacks certain features that you would normally find it the base Pytorch library, such as checkpoint saves, metric plotting.
Checkpoints are implemented as a model callback during training, and metric plotting is provided by Ignacio Oguiza at 'https://forums.fast.ai/t/plotting-metrics-after-learning/69937'.

Torchvision is another large component of the provided scripts as the library provides many useful utilities for manipulating images and tensors.

See 'www.fast.ai' for more details. 

#### Training and Validation
We trained the model on  a variety values for each of the hyperparameters and the training took place on a variety of lab machines. For the DiceLoss and JaccardLoss in particular, these were tested using the functions from the segmentation_models.pytorch[2] library. A dicussion on the Pytorch forums[4] helped to think of ways to test the DiceLoss.

For training accuracy, we measured if the pixels were the same between the prediction and target. For the validation accuracy, we used mean intersection over union, also called the jaccard index. We initially tried to do it manually but it turned out that that calculation was wrong so we ended up using the torchmetrics library[1] which has a prewritten intersection over union (jaccardindex) function to do this calculation.

The logs folder contains the logs for models. The names of the folders in the logs files that are prefixed with a room and pc number correspond to the lab machine tested. Note that many of the accuracies reported are not correct as we had faulty calculations for the validation mIoU.


#### Results
The final model we settled on was trained on 18000 images for 100 epochs with a batch size of 20. It had a training rate of 0.0001 with a Cross Entropy Loss function. We used RandomCrop as the cropping technique on 256 by 256 images. The model can be found under folder within the more_logs folder with the name model-18000-RandomCrop-256-20-00001-100-wCE. 

Here are some of the results of this model:
![Output Example](https://github.com/aichoudry/csc490-mapillary-vistas/blob/main/more_logs/model-18000-RandomCrop-256-20-00001-100-wCE/image_results/FYoB0UGu40k9nKurB8pucw.png?raw=true)
![Output Example](https://github.com/aichoudry/csc490-mapillary-vistas/blob/main/more_logs/model-18000-RandomCrop-256-20-00001-100-wCE/image_results/1pFw6qQwFQ__QdnYYB7YLA.png?raw=true)
![Outpuut Example](https://github.com/aichoudry/csc490-mapillary-vistas/blob/main/more_logs/model-18000-RandomCrop-256-20-00001-100-wCE/image_results/-C-x3xSPFIEjqbyVC5PRaQ.png?raw=true)


With this model, we achieved a validation mean intersection over union of about 37 percent[1]. This was done by using the jaccard index on the predictions and outputs one by one instead of as a batch. Using a batch provided a lower accuracy of about 15%. We believe that the 37% is more accuracte and using it unbatched provided the more accuracte result. This value is adequate considering our timeframe but still lower than any of the models on paperswithcode on this dataset although it is near the model with the lowest accuracy which is 39.7 percent[3]. 

The folders prefixed with _model_ _unet_ in the logs and more_logs folder contain the rest of the results of other models with different hyperparameters.

#### Conclusion
It appears that there are some fundamental difficulties with the architecture of our model. The choices of hyperparameters does not seem to offset the inherent flaws within UNet which shows that UNet is limited in the area of semantic segmentation. Making modifications to the UNet architecture instead of using a plain UNet model might yield better results. 


#### Contribution
**Abdurrahman Choudry:** Built the UNet model and created the main scripts for transforming, training and testing it. My focus was on researching and testing different loss functions (weighted) and data augmentation methods/image transformations we could do to minmize the time to train and memory usage while also maintain or increase accuracy  
**Matthew Ho Kin:** First through the research of the UNet model and then all the development and training of the Resnet-UNet model using the Fastai library.  
**Ryan D’Rosario:** Worked on getting the values for the hyperparameters to optimize the model.  
**Hanchao Ge:** Displaying output images with palette

#### References
[1] Nicki Skafte Detlefsen, Jiri Borovec, Justus Schock, Ananya Harsh, Teddy Koker, Luca Di Liello, Daniel Stancl, Changsheng Quan, Maxim Grechkin, & William Falcon. (2022). TorchMetrics - Measuring Reproducibility in PyTorch [Computer software]. https://doi.org/10.21105/joss.04101

]2] Iakubovskii, P. (2019). Segmentation Models Pytorch. In GitHub repository. GitHub. https://github.com/qubvel/segmentation_models.pytorch

[3] Semantic Segmentation on Mapillary val. (n.d.). paperswithcode.com; Meta. Retrieved December 04, 2023, from https://paperswithcode.com/sota/semantic-segmentation-on-mapillary-val

[4] Segmentation loss and metrics help ignore background. (2023, April 14). PyTorch Forums; Pytorch. Retrieved December 05, 2023, from https://discuss.pytorch.org/t/segmentation-loss-and-metrics-help-ignore-background/177519

[5] R. Takahashi, T. Matsubara and K. Uehara, "Data Augmentation Using Random Image Cropping and Patching for Deep CNNs," in IEEE Transactions on Circuits and Systems for Video Technology, vol. 30, no. 9, pp. 2917-2931, Sept. 2020, doi: 10.1109/TCSVT.2019.2935128.
