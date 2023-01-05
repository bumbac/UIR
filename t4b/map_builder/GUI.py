from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.config import Config
from functools import partial

BUTTON_SIZE = 75


class GUI(App):
    def __init__(self, x_size, y_size, model):
        self.x_size = y_size
        self.y_size = x_size
        self.model = model
        self.buttons = []
        self.layout = None
        Config.set('graphics', 'width', str(self.x_size * BUTTON_SIZE))
        Config.set('graphics', 'height', str((self.y_size + 1) * BUTTON_SIZE))
        super(GUI, self).__init__()

    def button_press(self, index, *args):
        self.model.grid_press(index)

    def select_mode(self, mode, *args):
        self.model.mode_selected(mode)

    def change_color(self, index, color):
        btn = self.buttons[index[0]][index[1]]
        btn.background_color = color

    def create_mode_button(self, mode, index, color):
        btn = Button(pos=(index * BUTTON_SIZE, 0), size_hint=(1 / self.x_size, 1 / (self.y_size + 1)), background_color=color)
        setattr(btn, "mode", mode)
        btn.bind(on_press=partial(self.select_mode, btn.mode))
        self.layout.add_widget(btn)

    def build(self):
        self.layout = FloatLayout()
        self.create_mode_button("wall", 0, (0.5, 0.5, 0.5, 1))
        self.create_mode_button("start", 1, (0, 1, 0, 1))
        self.create_mode_button("goal", 2, (1, 1, 0, 1))
        self.create_mode_button("sensor", 3, (1, 0, 0, 1))
        self.create_mode_button("save", 4, (0, 0, 1, 1))
        for i in range(self.x_size):
            self.buttons.append([])
            for j in range(self.y_size):
                btn = Button(pos=(i * BUTTON_SIZE, (j + 1) * BUTTON_SIZE), size_hint=(1 / self.x_size, 1 / (self.y_size + 1)))
                setattr(btn, "index", (i, j))
                btn.bind(on_press=partial(self.button_press, btn.index))
                self.buttons[i].append(btn)
                self.layout.add_widget(btn)
        return self.layout
