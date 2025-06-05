
[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@diego.machado/fine-tuning-vit-for-image-classification-with-hugging-face-48c4be31e367)

In this Repo we are solving an image classification use case (pet classification) using Hugging Face 🤗 (transformers, dataset and trainer) and Data Augmentation with Pytorch Transforms.

In the  [repo](https://github.com/diegulio/pytorch-breed-classification/blob/main/notebooks/breed_class_ViT_huggingface.ipynb)  you will be able to find notebooks for the same use case using ViT with huggingface and CNN with Pytorch Lightning.

# Introduction

In this post, we are going to leverage the capabilities of transformers applied to a computer vision problem powered by Huggingface 🤗.

Very often, transformers are used in natural language processing use cases. In the recent time, people started to use it also in computer vision applications such as image classification, getting comparable performance against the well-known Convolutional Neural Network models.

[Vision Transformer](https://arxiv.org/abs/2010.11929v2)  (ViT for short) is a model that employs a Transformer-like architecture over patches of the image. An image is split into fixed-size patches, each of them are then linearly embedded, position embeddings are added, and the resulting sequence of vectors is fed to a standard Transformer encoder

![Visual Transformer (ViT). Source: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://miro.medium.com/v2/resize:fit:642/0*ZNErH_i-EYHNYQIt.png)

ViT Architecture. Source: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

[Hugging Face](https://huggingface.co/)🤗 are the creators of the Transformers library, an open-source machine learning library. This library offers a range of pre-trained models based on transformer architectures, specifically designed for natural language processing (NLP) tasks. It enables users to effortlessly utilize, fine-tune, and deploy cutting-edge models in various applications like text classification, language translation, and text generation. In this case we will be using it for Vision.

> **Disclaimer**: In this post, I am not focused in explaining the theory behind ViT but the implementation. This is because there are plenty of very good theoretical information available in internet.

# **Tools**

These are the tools that we will use/learn:

1.  **Hugging Face**  🤗  
    a.  **Transfomers**: Simplifies the use of pre-trained transformer models for natural language processing tasks  
    b.  **Datasets: F**acilitates easy access, exploration, and manipulation of datasets for machine learning tasks.  
    c.  **Trainer**: Streamlines the training and fine-tuning of machine learning models, offering a cohesive framework for efficient experimentation and optimization.
2.  **Pytorch  
    a. Transforms: S**et of operations applied to data during loading, augmenting, or preprocessing in a flexible and modular manner for machine learning tasks.

# Prototyping

Along this section, we will be following the steps:

1.  Organize Folder according to 🤗 Datasets structure (ImageFolder)
2.  Creating and uploading the dataset to 🤗 Datasets
3.  Defining data augmentations
4.  Loading the model and processors with 🤗 Transformers
5.  Trainer with 🤗 Trainer

You are free skipping some of these steps as they might not be useful for your own use case.

## 1. Organize Folders according to 🤗 Datasets structure

Initially, assume that we have all our images of pets in a folder and a csv file with the labels for each of those images.

my_images/  
├── 000bec180eb18c7604dcecc8fe0dba07.jpg  
├── 001513dfcb2ffafc82cccf4d8bbaba97.jpg  
├── ...  
├── 007b5a16db9d9ff9d7ad39982703e429.jpg  
└── labels.csv

![](https://miro.medium.com/v2/resize:fit:417/1*jOhCo6QhpL7KtXrtByr2RA.png)

Sample of labels.csv

There are two ways to create a 🤗 Dataset, the straightforward one is with the “ImageFolder” type. This is a no-code solution for quickly creating an image dataset with several thousand images.

We just need to end with a structure like:

    my_dataset_repository/  
    ├── README.md  
    ├── train/classes/imgs  
    ├── validation/classes/imgs  
    └── test/classes/imgs

The following script will be very specific for my use case, unless you have an initial folder structure like mine, I would recommend you skipping this.

# Creating 🤗 Dataset structure  
from sklearn.model_selection import train_test_split  
from pathlib import Path  
  
# 1. Split Data   

    labels = pd.read_csv("labels.csv")  
    X = labels.id  
    y = labels.breed  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13, shuffle = True, stratify = y)  
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=13, shuffle = True, stratify = y_train)  
    train_data = pd.concat([X_train, y_train], axis = 1).reset_index(drop = True)  
    val_data = pd.concat([X_val, y_val], axis = 1).reset_index(drop = True)  
    test_data = pd.concat([X_test, y_test], axis = 1).reset_index(drop = True)  

  
  
  
# 2. make Dataset Dir and splits  

    DatasetDir = Path("PetClassification")  
    os.mkdir(DatasetDir)  
    os.mkdir(DatasetDir/"train")  
    os.mkdir(DatasetDir/"validation")  
    os.mkdir(DatasetDir/"test")  
      

# 3. make classes (breeds) dir  

    for breed in tqdm(train_data.breed.unique()):  
      os.mkdir(DatasetDir/"train"/breed)  
      os.mkdir(DatasetDir/"validation"/breed)  
      os.mkdir(DatasetDir/"test"/breed)  

  
# 4. moving images to split folder and breed  

    IMG_PATH = "path/to/current_images"  
    def make_split_folder(split_df, split):  
      # iterate over dataset  
      for idx, row in tqdm(split_df.iterrows(), total = len(split_df), desc = f"Making {split} folder"):  
        img_name, breed = row  
        # copy files  
        shutil.copyfile(f"{os.path.join(IMG_PATH, img_name)}.jpg", f"{DatasetDir/split/breed/img_name}.jpg")  
      
    make_split_folder(train_data, "train")  
    make_split_folder(val_data, "validation")  
    make_split_folder(test_data, "test")

Running this we will end with a “ImageFolder” structure. For more information about this you can see the  [documentation](https://huggingface.co/docs/datasets/image_dataset).

## 2. Creating and uploading the dataset to 🤗 Datasets

Having the “ImageFolder” structure, the rest is very straightforward thanks to 🤗 Datasets Library.

    from datasets import load_dataset  

# create dataset  

    pet_dataset = load_dataset("imagefolder", data_dir = "PetClassification")  

  
# uploading it to Hugging Face Hub  
  
# Push to Hub  

    notebook_login() # for notebooks  
    pet_dataset.push_to_hub("Diegulio/PetClassification")  

And that’s it. Now you will be able to see it in your datasets. Note that you have to login with your username and with an access token. You can create an access token in  _User->Setting->Access Tokens->New Token_

![](https://miro.medium.com/v2/resize:fit:700/1*kfeQ1-SSg92wZelYNpfnlA.png)

Dataset in Hub

You can vary some parameters, as for example make this dataset private, or creating a model card.

To prove that you have loaded your image accordingly, you can:

# Load dataset (now directly from Hub)  

    pet_dataset = load_dataset("Diegulio/PetClassification")  
      
    labels = pet_dataset["train"].features["label"].names  
      
    idx2label = {idx: label for idx, label in enumerate(labels)}  
    label2idx = {label: idx for idx, label in enumerate(labels)}  
      
    random_idx = random.randint(0,len(pet_dataset['train']))  
    print(f"Breed: {idx2label[pet_dataset['train'][random_idx]['label']]}")  
    pet_dataset['train'][random_idx]['image']

![](https://miro.medium.com/v2/resize:fit:341/1*5dAxU6F4UIurA7pmcOwYuw.png)

Random Image from dataset

## 3. Defining data augmentations

Data augmentation is a technique used in machine learning and computer vision to artificially increase the diversity of a training dataset by applying various transformations to the existing data. The goal is to create additional variations of the original data without collecting new samples, which can help improve the generalization and robustness of a machine learning model.

Here, we will use Pytorch transform library to accomplish it. Even though, there is an important point to discuss before:

Each transformer model has an special way in which the data should pass through. In case of Vision Transformers, each one has a different image sizing, normalizing, scaling, etc. All this information is in the  _processor_  (we download it from the Hub just like the model). People often pass all the images through the processor with a map function like this:

    from transformers import ViTImageProcessor  
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224', return_tensors = 'pt')  
      
    def apply_processor(example):  
      example['pixel_values'] = processor(example['image'].convert("RGB"), return_tensors="pt").pixel_values.squeeze()  
      return example  
      
    processed_dataset = pet_dataset.map(apply_processor)

You can add the augmentations inside the apply_processor function as well. The thing is that for some reason the map function make the training slower! I have not be able to figured it out why (I asked in the  [Hugging Face Forum](https://discuss.huggingface.co/t/using-map-take-7-2x-times-longer-than-set-transform/62285), no answers 😢)

**Solution**: set_transform(). Here I will show you how I surpass it. First I define the augmentations to be applied:

  

    from transformers import ViTImageProcessor  
    from torchvision.transforms import v2  
      
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224', do_rescale = False, return_tensors = 'pt')  
      
    image_mean, image_std = processor.image_mean, processor.image_std  
    size = processor.size["height"]  
      
    normalize = v2.Normalize(mean=image_mean, std=image_std)  
      
    train_transform = v2.Compose([  
          v2.Resize((processor.size["height"], processor.size["width"])),  
          v2.RandomHorizontalFlip(0.4),  
          v2.RandomVerticalFlip(0.1),  
          v2.RandomApply(transforms=[v2.RandomRotation(degrees=(0, 90))], p=0.5),  
          v2.RandomApply(transforms=[v2.ColorJitter(brightness=.3, hue=.1)], p=0.3),  
          v2.RandomApply(transforms=[v2.GaussianBlur(kernel_size=(5, 9))], p=0.3),  
          v2.ToTensor(),  
          normalize  
          #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))  
     ])  
      
    test_transform = v2.Compose([  
        v2.Resize((processor.size["height"], processor.size["width"])),  
        v2.ToTensor(),  
        normalize  
    ])

Note that I apply by myself the main transformations applied in the processor (Resize, normalize and rescale). If you are asking, the rescale part is inside the ToTensor() function.

Be careful if we select another model, the processor’s attributes could be different.

Then I create the processing functions and apply set_transform(). The difference between the map function and the set_transform is that the set_transform apply the function in-fly! That means that it apply it every time __get_item__ is called behind scenes.

    def train_transforms(examples):  
        examples['pixel_values'] = [train_transform(image.convert("RGB")) for image in examples['image']]  
        return examples  
      
    def test_transforms(examples):  
        examples['pixel_values'] = [test_transform(image.convert("RGB")) for image in examples['image']]  
        return examples  

  
# Set the transforms  

    pet_dataset['train'].set_transform(train_transforms)  
    pet_dataset['validation'].set_transform(test_transforms)  
    pet_dataset['test'].set_transform(test_transforms)

For me, is counterintuitive that the map function make the training slower because we are applying the transform before! D: If you know the answer, let me know the truth!!

Note that we use the  _test_transform_  in  _test_  and  _validation_ sets, and it is not the same as the training one. This is because at inference time we will just want to predict the image that we were asked for, not the rotated one! 😵‍💫

Sometime is difficult to imagine the effects of your augmentation code to your images. There is a good application that make this easier! (shameless self-promotion). You can interactively test and see different augmentation!

-   Application link:  [https://pytorch-transform-illustrations.streamlit.app](https://pytorch-transform-illustrations.streamlit.app/)
-   Repo:  [https://github.com/diegulio/pytorch-transform-illustrations/tree/main](https://github.com/diegulio/pytorch-transform-illustrations/tree/main)

![](https://miro.medium.com/v2/resize:fit:442/1*c8gFkn1BdTBLjUwb3VMsvQ.gif)

Transforms Illustrated

## 4. Loading the model and processors with 🤗 Transformers

Now we will define and load all stuff that we need to train, this is: model, processor, metric function and data collator.

First model and processors. Today we are using the google ViT model.

    from transformers import ViTForImageClassification  
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224',  
                                                      id2label=idx2label,  
                                                      label2id=label2idx,  
                                                      ignore_mismatched_sizes=True)  
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224', do_rescale = False, return_tensors = 'pt')

Then the metric function.

    from sklearn.metrics import accuracy_score  
    import numpy as np  
      
    def compute_metrics(eval_pred):  
        predictions, labels = eval_pred  
        predictions = np.argmax(predictions, axis=1)  
        return dict(accuracy=accuracy_score(predictions, labels))

And finally the data collator. This is because we need to form batches from our data, with data collators we tell to the libraries how to do it. Also sometimes our data could have a lot of features, like the images, the labels, the images augmented, the pixel values or some other metadata. With the data collator we can just select the ones we need. Lastly, we also can apply some operations like augmentations or paddings in the data collator.

Sometimes data collator is not necessary. In our case, is straightforward.

    def collate_fn(examples):  
        pixel_values = torch.stack([example["pixel_values"] for example in examples])  
        labels = torch.tensor([example["label"] for example in examples])  
        return {"pixel_values": pixel_values, "labels": labels}

## 5. Trainer with 🤗 Trainer

Thanks to all the steps we have gone through, this is the easy part. We just need to use 🤗[Trainer](https://huggingface.co/docs/transformers/main_classes/trainer).

    from transformers import TrainingArguments, Trainer  
      
    metric_name = "accuracy"  

  
# Define Train Parameters  

    args = TrainingArguments(  
        f"breed-classification",  
        use_cpu = False,  
        evaluation_strategy="steps",  
        logging_steps = 100,  
        learning_rate=2e-5,  
        per_device_train_batch_size=64,  
        per_device_eval_batch_size=64,  
        num_train_epochs=15,  
        weight_decay=0.01,  
        load_best_model_at_end=True,  
        metric_for_best_model=metric_name,  
        logging_dir='logs',  
        remove_unused_columns=False,  
        push_to_hub = True,  
        hub_model_id = "MyPetModel"  
    )  
      

# Train  

    trainer = Trainer(  
        model,  
        args,  
        train_dataset=pet_dataset['train'],  
        eval_dataset=pet_dataset['validation'],  
        data_collator=collate_fn,  
        compute_metrics=compute_metrics,  
        tokenizer=processor,  
    )

We will end with something like this:

![](https://miro.medium.com/v2/resize:fit:487/1*2aRqENitUHqOzqPgCIRF0w.png)

Final results

Note that we push the model to the hub in the TrainingArguments! You can see more options in the  [documentation](https://huggingface.co/docs/transformers/v4.35.2/en/main_classes/trainer#transformers.TrainingArguments)




