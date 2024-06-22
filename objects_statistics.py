import pickle
import os
import numpy as np

class ObjectStatistics:
    def __init__(self, file_path):
        self.file_path = file_path
        self.image_data = self.load_pickle()
        self.object_counts = self.count_objects()

    def load_pickle(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        
        with open(self.file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    def count_objects(self):
        object_counts = []

        for image_key in self.image_data:
            objects = self.image_data[image_key].get('objects', [])
            object_counts.append(len(objects))

        return object_counts

    def calculate_statistics(self):
        min_objects = np.min(self.object_counts)
        max_objects = np.max(self.object_counts)
        mean_objects = np.mean(self.object_counts)
        total_objects = np.sum(self.object_counts)

        return min_objects, max_objects, mean_objects, total_objects

    def print_statistics(self):
        min_objects, max_objects, mean_objects, total_objects = self.calculate_statistics()
        print(f"Minimum number of objects: {min_objects}")
        print(f"Maximum number of objects: {max_objects}")
        print(f"Mean number of objects: {mean_objects:.2f}")
        print(f"Total number of objects: {total_objects}")

    def run(self):
        self.print_statistics()

if __name__ == '__main__':
    pickle_file_path = '/media/elena/Elements/Dataset/impara-dataset/gt.pickle'  # Replace with your pickle file path
    stats = ObjectStatistics(pickle_file_path)
    stats.run()
