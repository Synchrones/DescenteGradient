from manim import *
from manim.opengl import *
import pathlib

class Test(Scene):
    def construct(self):
        hello_world = Tex("Hello World!").scale(3)
        self.play(Write(hello_world))
        self.play(self.camera.animate.set_euler_angles(theta=-10*DEGREES, phi=50*DEGREES))
        self.play(FadeOut(hello_world))
        surface = OpenGLSurface(lambda u, v: (u, v, u*np.sin(v) + v*np.cos(u)), u_range=(-3, 3), v_range=(-3, 3))
        surface_mesh = OpenGLSurfaceMesh(surface)
        self.play(Create(surface_mesh))
        self.play(FadeTransform(surface_mesh, surface))
        self.wait()
        light = self.camera.light_source
        self.play(light.animate.shift((0, 0, -20)))
        self.play(light.animate.shift((0, 0, 10)))
        self.play(self.camera.animate.set_euler_angles(theta=60*DEGREES))
        self.interactive_embed()



class Test2(Scene):
    def construct(self):
        surface = OpenGLSurface(lambda u, v: (u, v, np.cos(u) * np.sin(v)),
                                u_range=(-5, 5),
                                v_range=(-5, 5),
                                color=BLUE
                                )
        self.play(Create(surface))
        # surface.set_color_by_code("color = vec4(point.x, point.y, point.z, 1);")
        surface.reload_shader_wrapper()
        surface.get_shader_wrapper().shader_folder = "derive_x"
        self.interactive_embed()


class Test3(Scene):
    def construct(self):
        axes = ThreeDAxes()
        self.add(axes)
        surface = Surface(lambda u, v: (u, v, np.cos(u) * np.sin(v)),
                                u_range=(-5, 5),
                                v_range=(-5, 5),
                                fill_color=BLUE,
                                resolution=(20, 20),
                                fill_opacity = 1
                                )
        self.play(Create(surface))
        # surface.set_fill_by_value(axes, [BLUE, RED])
        self.interactive_embed()
