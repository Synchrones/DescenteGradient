import random

from manim import *
from pyglet.window import *

depart = []
derives = [lambda x: float((np.cos(x) * x - np.sin(x)) / x**2)]
nouvelles_coords = []
point = None
axes = None

def fnc(x):
    return float(np.sin(x) / x)

class DescenteGradient(Scene):

    def construct(self):
        global depart
        global derives
        global nouvelles_coords
        global point
        global axes
        axes = Axes(x_range=(-10, 10), y_range=(-1, 1))
        labels = axes.get_axis_labels(x_label="x", y_label="y")
        courbe = axes.plot(fnc, color=YELLOW)
        depart = [random.randint(-5, 5)]
        nouvelles_coords = depart[:]
        point = Dot(point=axes.coords_to_point(depart[0], fnc(depart[0])))
        self.play(Create(axes, run_time=3), Create(labels), Create(courbe, run_time=3))
        self.add(point)

        self.interactive_embed()



    def on_key_press(self, symbol, modifiers):
        global depart
        global derives
        global nouvelles_coords
        global point
        global axes
        if symbol == key.SPACE:
            print(depart)
            pentes = [derives[i](depart[i]) for i in range(len(derives))]
            for var in range(len(pentes)):
                nouvelles_coords[var] = depart[var] - pentes[var] * 2
                print(nouvelles_coords, depart, pentes)
                try:
                    courbe_partielle = axes.plot(fnc, x_range=[min(depart[0], nouvelles_coords[0]),
                                                               max(depart[0], nouvelles_coords[0]),
                                                               (abs(depart[0]) + abs(nouvelles_coords[0])) / 50])
                    if nouvelles_coords[0] - depart[0] > 0:
                        self.play(MoveAlongPath(point, courbe_partielle), run_time=1, rate_func=smooth)
                    else:
                        self.play(MoveAlongPath(point, courbe_partielle.reverse_points()), run_time=0.5, rate_func=smooth)
                except ValueError:
                    self.remove(point)
                    point = Dot(point=axes.coords_to_point(nouvelles_coords[0], fnc(nouvelles_coords[0])))
                    self.add(point)
                depart = nouvelles_coords[:]