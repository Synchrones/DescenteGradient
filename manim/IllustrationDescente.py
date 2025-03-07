import random

import numpy as np
from manim import *
from pyglet.window import *
from manim.opengl import *


suivant = False
change_surface = False
shift = False

# utilisé pour mettre en pause le programme en attendant
def attendre_entree():
    return suivant

# appuyer sur espace passe à l'itération suivante de la descente
class DescenteGradient(Scene):
    def construct(self):
        def fnc(x):
            return np.sin(x) / x

        global suivant

        derives = [lambda x: (np.cos(x) * x - np.sin(x)) / x ** 2]
        axes = Axes(x_range=(-10, 10), y_range=(-1, 1))
        labels = axes.get_axis_labels(x_label="x", y_label="y")
        courbe = axes.plot(fnc, color=YELLOW)
        depart = [random.uniform(-5, 5)]
        nouvelles_coords = depart[:]
        point = Dot(point=axes.coords_to_point(depart[0], fnc(depart[0])))
        self.play(Create(axes, run_time=3), Create(labels), Create(courbe, run_time=3))
        self.add(point)
        """
        label_coords = always_redraw(lambda: Text(f"Coordonnées du point : "
                                                  f"{[round(float(coord), 2) for coord in axes.coords_to_point(*point.get_center()[:-1])[:-1]]}",
                                                  font_size=30)
                                     .to_edge(DR))
        self.add(label_coords)
        """
        termine = False
        while not termine:
            termine = True
            self.wait(100000, attendre_entree)  # est passé quand la fonction on_key_press est appelée
            suivant = False
            print(depart)
            pentes = [derives[i](depart[i]) for i in range(len(derives))]
            for var in range(len(pentes)):
                nouvelles_coords[var] = depart[var] - pentes[var] * 2
                if abs(pentes[var]) > 0.01:
                    termine = False
                try:
                    courbe_partielle = axes.plot(fnc, x_range=[min(depart[0], nouvelles_coords[0]),
                                                               max(depart[0], nouvelles_coords[0]),
                                                               (abs(depart[0]) + abs(nouvelles_coords[0])) / 50])
                    if nouvelles_coords[0] - depart[0] > 0: # définie le sens de déplacement
                        self.play(MoveAlongPath(point, courbe_partielle), run_time=0.5, rate_func=smooth)
                    else:
                        self.play(MoveAlongPath(point, courbe_partielle.reverse_points()), run_time=0.5,
                                  rate_func=smooth)
                except ValueError: # problèmes pour créer la courbe si le déplacement est trop petit
                    self.play(point.animate.move_to(axes.coords_to_point(nouvelles_coords[0], fnc(nouvelles_coords[0]))),
                              run_time=0.4, rate_func=smooth)
                depart = nouvelles_coords[:]

        self.interactive_embed()

    def on_key_press(self, symbol, modifiers):
        global suivant
        if symbol == key.SPACE:
            suivant = True


class Descente3D(Scene):
    def construct(self):
        def fnc(u, v):
            return np.cos(u) * np.sin(v)
        derives = [lambda u, v : - np.sin(u) * np.sin(v), lambda u, v: np.cos(u) * np.cos(v)]


        def deplace_point(pt, alpha):
            u = interpolate(depart[0], nouvelles_coords[0], alpha)
            v = interpolate(depart[1], nouvelles_coords[1], alpha)
            x, y, z = np.array([u, v, fnc(u, v)])
            pt.move_to([x, y, z])

        global suivant
        global change_surface
        surfaces = []
        compteur = 0
        self.camera.set_euler_angles(phi=45 * DEGREES, theta=45 * DEGREES)
        axes = ThreeDAxes(x_range=[-10, 10], y_range=[-10, 10], z_range=[-10, 10])
        labels = axes.get_axis_labels(x_label="x", y_label="y", z_label="z")
        self.play(Create(axes), Create(labels), run_time=4)

        surface = OpenGLSurface(lambda u, v: (u, v, fnc(u, v)),
                                u_range=(-5, 5),
                                v_range=(-5, 5),
                                color=BLUE,
                                shadow=0.2
                                )
        surface.reload_shader_wrapper()
        surface.get_shader_wrapper().shader_folder = "gradient_z"
        surfaces.append(surface)

        surface_derive_x = OpenGLSurface(lambda u, v: (u, v, fnc(u, v)),
                                u_range=(-5, 5),
                                v_range=(-5, 5),
                                color=BLUE,
                                shadow=0.2
                                )
        surface_derive_x.should_render = False
        self.add(surface_derive_x)
        surface_derive_x.reload_shader_wrapper()
        surface_derive_x.get_shader_wrapper().shader_folder = "derive_x"
        surfaces.append(surface_derive_x)

        surface_derive_y = OpenGLSurface(lambda u, v: (u, v, fnc(u, v)),
                                         u_range=(-5, 5),
                                         v_range=(-5, 5),
                                         color=BLUE,
                                         shadow=0.2
                                         )
        surface_derive_y.should_render = False
        self.add(surface_derive_y)
        surface_derive_y.reload_shader_wrapper()
        surface_derive_y.get_shader_wrapper().shader_folder = "derive_y"
        surfaces.append(surface_derive_y)

        self.play(Create(surface))

        depart = [random.uniform(-3, 3), random.uniform(-3, 3)]
        nouvelles_coords = depart[:]
        point = Dot3D(point=[depart[0], depart[1], fnc(depart[0], depart[1])],
                      color=RED,
                      radius=0.08)
        self.add(point)
        termine = False
        while not termine:
            self.wait(100000, attendre_entree)  # est passé quand la fonction on_key_press est appelée
            suivant = False
            if change_surface:
                change_surface = False
                surfaces[compteur % 3].should_render = False
                if shift:
                    compteur -= 1
                else:
                    compteur += 1
                surfaces[compteur % 3].should_render = True
                continue

            termine = True
            print(depart)
            pentes = [derives[i](*depart) for i in range(len(derives))]
            for var in range(len(pentes)):
                nouvelles_coords[var] = depart[var] - pentes[var] * 1
                if abs(pentes[var]) > 0.001:
                    termine = False
            #self.play(MoveAlongPath(point, creer_courbe_partielle(depart, nouvelles_coords, fnc)), run_time=0.5)
            self.play(UpdateFromAlphaFunc(point, deplace_point), run_time=1) # TODO : manière plus opti de bouger point? (gros délais)
            depart = nouvelles_coords[:]

        self.interactive_embed()

    def on_key_press(self, symbol, modifiers):
        global suivant
        global change_surface
        global shift
        if symbol == key.SPACE:
            suivant = True
        if symbol == key.TAB:
            suivant = True
            change_surface = True
        if symbol == key.LSHIFT:
            shift = True

    def on_key_release(self, symbol, modifiers):
        global shift
        if symbol == key.LSHIFT:
            shift = False




def creer_courbe_partielle(depart, arrive, fnc):
    u_fnc = lambda t: depart[0] + (arrive[0] - depart[0]) * t
    v_fnc = lambda t: depart[1] + (arrive[1] - depart[1]) * t
    return ParametricFunction(lambda t : np.array([u_fnc(t), v_fnc(t), fnc(u_fnc(t), v_fnc(t))]))