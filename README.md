# Diabetic Retinopathty Detection
Diabetic retinopathy (DR) is a serious ocular condition that can significantly compromise the vision of individuals with diabetes. In this study, we utilized multiple detection models based on deep convolutional neural networks and compared their performance. All models were trained using the IDRiD dataset. Additionally, we employed data augmentation, transfer learning and ensemble learning techniques, enabling our best model to achieve an accuracy of 84%. Ultimately, our findings suggest that the size of the dataset is the most critical factor influencing detection performance.

## Content
* **Dataset:** Utilizes the Indian Diabetic Retinopathy Image Dataset (IDRiD) comprising 516 images.
* **Input Pipeline:** Employs TFRecord for efficient data loading and processing.
* **Image Preprocessing:** Involves cutting black edges, resizing, and normalization. Implements various data augmentation techniques to enhance model robustness.
* **Architecture Families:** Leverages well-known CNN architectures such as MobileNet, EfficientNet, and DenseNet.
* **Metrics:** Evaluates performance using binary accuracy and confusion matrix analyses.
* **Transfer Learning:** Applies transfer learning in model variants EfficientNetB3 and DenseNet201.
* **Ensemble Learning:** Constructs composed model from MobileNetV2, EfficientNetB0 with Augmentation and pre-trained EfficientNetB3, DenseNet201.
* **Hyperparameter Tuning:** Enhance the performance of the individual model variants.
* **Deep Visualization Techniques:** Use methods Grad-CAM, Guided Backpropagation, Guided Grad-CAM, and Integrated Gradients to interpret model decisions.


##  Results
Test accuracies for different model variants:

| **Architecture Familiy** | **Model Variants** | **Augmentation** | **Transfer Learning** | **Test Accuracy** |
|--------------------------|--------------------|------------------|-----------------------|-------------------|
| MobileNet                | MobileNetV2        | False            | False                 | 0.75              |
| "                        | "                  | True             | False                 | 0.80              |
| EfficientNet             | EfficientNetB0     | False            | False                 | 0.73              |
| "                        | "                  | True             | False                 | 0.80              |
| "                        | EfficientNetB3     | True             | True                  | 0.84              |
| DenseNet                 | DenseNet201        | True             | True                  | 0.79              |

The test accuracy of the composed model from MobileNetV2, EfficientNetB0 with Augmentation and pre-trained EfficientNetB3, DenseNet201 results in: 0.80

## How to run the Code?
* **Step 0:** Download the Indian Diabetic Retinopathy Image Dataset (IDRiD) from [here](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid).
* **Step 1:** Adjust the paths of `load.img_dir` and `load.csv_dir` in `config.gin` to the images/labels folder of the data directory, the IDRiD dataset.
* **Step 2.a:** Configure the program manually to be able to use the full range of functions.
  * **Step 2.a.a:** Train a model
    * **Step 2.a.a.1:** Set the parameter `main_logic.model_names` in `config.gin` to `['MobileNetV2']`/`['EfficientNetB0']`/`['EfficientNetB3_pretrained']`/`['DenseNet201_pretrained']` to train the model variant MobileNetV2/EfficientNetB0/EfficientNetB3/DenseNet201. The latter two model variants use a transfer learning apprach.
    * **Step 2.a.a.2:** Set the parameter `main_logic.mode` in `config.gin` to `'train'`
    * **Step 2.a.a.3:** Set the remaining parameters in `config.gin`.
    * **Step 2.a.a.4:** Run `run.sh`.
  * **Step 2.a.b:** Evaluate a model
    * **Step 2.a.b.1:** Set the parameter `main_logic.model_names` in `config.gin` to `['MobileNetV2']`/`['EfficientNetB0']`/`['EfficientNetB3_pretrained']`/`['DenseNet201_pretrained']`/`['ComposedModel']` to evaluate the model variant MobileNetV2/EfficientNetB0/EfficientNetB3/DenseNet201 or a composed model. The model you want to evaluate has to be trained already!
    * **Step 2.a.b.2:** Set the parameter `main_logic.mode` in `config.gin` to `'evaluate'`
    * **Step 2.a.b.3:** Set the remaining parameters in `config.gin`.
    * **Step 2.a.b.4:** Run `run.sh`.
  * **Step 2.a.c:** Tune a model
    * **Step 2.a.c.1:** Set the parameter `main_logic.model_names` in `config.gin` to `['MobileNetV2']`/`['EfficientNetB0']`/`['EfficientNetB3_pretrained']`/`['DenseNet201_pretrained']` to tune the model variant MobileNetV2/EfficientNetB0/EfficientNetB3/DenseNet201.  The model you want to tune has to be trained already!
    * **Step 2.a.c.2:** Set the parameter `main_logic.mode` in `config.gin` to `'tune'`
    * **Step 2.a.c.3:** Set the remaining parameters in `config.gin` and `tune_wandb.py`.
    * **Step 2.a.c.4:** Run `run.sh`.
  * **Step 2.a.d:** Create a composed model
    * **Step 2.a.d.1:** E.g., set the parameter `main_logic.model_names` in `config.gin` to `['MobileNetV2', 'EfficientNetB0', 'EfficientNetB3_pretrained', 'DenseNet201_pretrained']` to create a composed model from the model variants MobileNetV2, EfficientNetB0, EfficientNetB3, DenseNet201.  The models you want to compose have to be trained already.
    * **Step 2.a.d.2:** Set the parameter `main_logic.mode` in `config.gin` to `'create_ensemble_model'`
    * **Step 2.a.d.3:** Set the remaining parameters in `config.gin`.
    * **Step 2.a.d.4:** Run `run.sh`.
* **Step 2.b:** Use a Quickstart configuration to train or evaluate the EfficientNetB3 model and get the best results we achieved in this project.
  * **Step 2.b.a:** Run `run_quickstart_train_efficientnetb3.sh` to train the EfficientNetB3 model variant.
  * **Step 2.b.b:** Run `run_quickstart_evaluate_efficientnetb3.sh` to evaluate the EfficientNetB3 model variant you trained in Step2b.a. 

Note: Sequences of letters, e.g. 2.a, 2.b, represent alternatives. Number sequences, e.g. 0., 1., represent orders!
