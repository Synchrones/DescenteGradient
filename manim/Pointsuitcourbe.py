from manim import *
class PointQuiSuitCourbe(Scene):
    def construct(self):
        AXY= Axes(x_range=(-0,2,0.5),y_range=(5,14))
        labels = AXY.get_axis_labels(x_label="x", y_label="y") #nomme les axes x et y
        def fonction(x): #fonction qu'on définie
            return 5*x**2-21*x**3+12*x**4+10+2*x
        Courbe = AXY.plot(fonction, color = YELLOW) 
        Point = Dot(point = ORIGIN) #création du point 
        x_start = -0.2 #point de départ
        x_min = 1.08512 #point d'arrivé (objectif minimum)
        partial_curve = AXY.plot(fonction, x_range=[x_start, x_min]) #intervalle de parcours du point
        self.play(Create(AXY, run_time = 3) ,Create(labels),Create(Courbe,run_time = 3))
        self.play(MoveAlongPath(Point, partial_curve), run_time = 3, rate_func=smooth) #animation du point le long de l'intervalle qu'on veut
        self.wait(3)
        self.interactive_embed()
        