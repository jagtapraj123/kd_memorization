import pandas as pd
import os
import random
import sys
import argparse


def data_init(
    dataset_loc, train_output_file, num_images_threshold, test_output_file=None
):
    classes = os.listdir(dataset_loc)

    train_data = pd.DataFrame({"image": [], "label": [], "class_name": []})
    train_data = train_data.astype({"image": str, "label": int, "class_name": str})

    if test_output_file is not None:
        test_data = pd.DataFrame({"image": [], "label": [], "class_name": []})
        test_data = test_data.astype({"image": str, "label": int, "class_name": str})

    for i, c in enumerate(classes):
        classpath = "{}/{}/".format(dataset_loc, c)
        if os.path.isdir(classpath):
            # class_name = c.removeprefix("pins_")
            class_name = c.split("pins_")[1]
            images = os.listdir(classpath)
            assert (
                len(images) >= num_images_threshold
            ), 'Number of images for class "{}" are not above num_images_threshold ({})'.format(
                c, num_images_threshold
            )

            random.shuffle(images)
            train_images = images[:num_images_threshold]
            test_images = images[num_images_threshold:]
            # train_images = random.sample(images, num_images_threshold)

            temp_list = []
            for si in train_images:
                temp_list.append(
                    {"label": i, "class_name": class_name, "image": classpath + si}
                )
            train_data = pd.concat(
                [train_data, pd.DataFrame(temp_list)], ignore_index=True
            )

            if test_output_file is not None:
                temp_list = []
                for si in test_images:
                    temp_list.append(
                        {"label": i, "class_name": class_name, "image": classpath + si}
                    )
                test_data = pd.concat(
                    [test_data, pd.DataFrame(temp_list)], ignore_index=True
                )

    train_data.to_csv(train_output_file, index=False)
    if test_output_file is not None:
        test_data.to_csv(test_output_file, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Data File Creator")
    parser.add_argument(
        "-n",
        type=int,
        help="number of images to keep in training set per class",
        default=5,
    )
    parser.add_argument(
        "-d",
        type=str,
        help="dataset root path",
        default="105_classes_pins_dataset",
    )
    parser.add_argument(
        "-otrain",
        type=str,
        help="output path for train data csv",
        default="train_image_data.csv",
    )
    parser.add_argument(
        "-otest",
        type=str,
        help="output path for test data csv",
        default="test_image_data.csv",
    )
    args = parser.parse_args()

    data_init(args.d, args.otrain, args.n, args.otest)
