import os
from gallery import Gallery
from double_oracle import double_oracle
import numpy as np
# from reference import double_oracle as reference_double_oracle


if __name__ == '__main__':
    for filename in os.listdir("maps/"):
        gallery = Gallery.gallery_from_file("maps/"+filename)
        # comment this to not plot the gallery at the start of the run
        gallery.plot_gallery()
        sensor_strategy, sensors, path_strategy, paths = double_oracle(gallery)
        full_sensor_prob = np.zeros(gallery.num_sensors)
        for sensor_prob, sensor_index in zip(sensor_strategy, sensors):
            full_sensor_prob[sensor_index] = sensor_prob
        for path_prob, path in zip(path_strategy, paths):
            if path_prob > 1e-6:
                gallery.plot_path(path, label=path_prob)
        gallery.plot_gallery(sensor_prob=full_sensor_prob)
