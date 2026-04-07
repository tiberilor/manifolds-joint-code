import os
import pickle


def create_categories_pickle(dataset_dir):
    """
    Creates a categories.pkl file in the dataset directory.
    The file maps each category (subdirectory in dataset_dir) to a unique index.
    The categories are sorted alphabetically.

    Args:
        dataset_dir (str): Path to the dataset folder containing category subfolders.
    """
    # List all entries in dataset_dir and filter to directories.
    categories = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    # Sort categories alphabetically.
    categories = sorted(categories)

    # Create a dictionary mapping category names to indices.
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}

    # Save the dictionary as categories.pkl in the dataset_dir.
    output_path = os.path.join(dataset_dir, "categories.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(cat_to_idx, f)

    print(f"Saved category mapping for {len(categories)} categories to {output_path}")


if __name__ == '__main__':
    # Update this path to your dataset directory.
    dataset_dir = "path/to/dataset"  # Dataset root containing one subdirectory per category.
    create_categories_pickle(dataset_dir)
