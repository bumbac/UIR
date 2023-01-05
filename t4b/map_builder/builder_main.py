from map_builder.GUI import GUI
from map_builder.map_model import MapModel

if __name__ == '__main__':
    x_size = 3
    y_size = 6
    map_model = MapModel(x_size, y_size)
    gui = GUI(x_size, y_size, map_model)
    map_model.set_gui(gui)
    gui.run()
