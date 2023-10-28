import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Set the path to the dataset folder
# dataset_path = 'dataset/train'
dataset_path = 'dataset/test'


# Task 1: Plot a bar graph showing the number of images in each class
def plot_class_distribution():
    class_counts = {}
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            class_counts[class_name] = len(images)

    plt.bar(class_counts.keys(), class_counts.values())
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.show()


# Function to load random images from each class
def load_random_images(num_images_per_class):
    random_images = []
    class_labels = []

    # Create a list of class folders
    class_folders = ["angry", "bored", "focused", "neutral"]

    # Track the number of images selected for each class
    class_image_counts = {class_name: 0 for class_name in class_folders}

    while len(random_images) < num_images_per_class:
        for class_name in class_folders:
            class_path = os.path.join(dataset_path, class_name)
            images = os.listdir(class_path)

            if class_image_counts[class_name] < 6:
                # Ensure at least one image from each class
                selected_image = random.choice(images)
                random_images.append(os.path.join(class_path, selected_image))
                class_labels.append(class_name)
                class_image_counts[class_name] += 1
            else:
                # Randomly select additional images from classes
                num_images_to_select = min(num_images_per_class - len(random_images), len(images))
                selected_images = random.sample(images, num_images_to_select)
                random_images += [os.path.join(class_path, img) for img in selected_images]
                class_labels += [class_name] * num_images_to_select

    return random_images, class_labels


# Task 2: Display 25 random images in a 5x5 grid with class labels and grid axis labeling
def display_random_images(random_images, class_labels):
    fig, axes = plt.subplots(5, 5, figsize=(8.5, 11))
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, hspace=0.2, wspace=0.2)

    for i, ax in enumerate(axes.flat):
        img = mpimg.imread(random_images[i])
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(class_labels[i])

        # Add class label at the bottom
        ax.set_xlabel(class_labels[i], labelpad=10)

    plt.show()


# Task 3: Plot a histogram of pixel intensities
def plot_pixel_intensity_histogram(random_images):
    plt.figure(figsize=(10, 6))

    for img_path in random_images:
        img = mpimg.imread(img_path)
        for i, color in enumerate(['Red', 'Green', 'Blue']):
            pixel_values = img[:, i].flatten()
            plt.hist(pixel_values, bins=50, color=color.lower(), alpha=0.5)

    plt.title("Pixel Intensity Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Number of random images to display
    num_random_images = 25

    # Load random images and class labels from each class
    random_images, class_labels = load_random_images(num_random_images)

    # Task 1: Plot the class distribution
    plot_class_distribution()

    # Task 2: Display 25 random images in a 5x5 grid with class labels and grid axis labeling
    display_random_images(random_images, class_labels)

    # Task 3: Plot the pixel intensity histogram
    plot_pixel_intensity_histogram(random_images)
