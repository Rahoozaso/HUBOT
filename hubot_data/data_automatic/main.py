from modules.dataset_generator import generate_dataset, save_dataset

if __name__ == "__main__":
    dataset = generate_dataset()
    save_dataset(dataset)
