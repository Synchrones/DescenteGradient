#version 330

#include ../include/camera_uniform_declarations.glsl

in vec3 point;
in vec3 du_point;
in vec3 dv_point;
in vec4 color;

out vec3 xyz_coords;
out vec3 v_normal;
out vec4 v_color;

#include ../include/position_point_into_frame.glsl
#include ../include/get_gl_Position.glsl
#include ../include/get_rotated_surface_unit_normal_vector.glsl

vec4 red_green_gradient(float value, float min_value, float max_value){
    float red;
    float green;
    if (value < (max_value + min_value) / 2) {
        red = 2 * (value - min_value) / (max_value - min_value);
        green = 1;
    }
    else {
        red = 1;
        green = 2 * (max_value - value) / (max_value - min_value);
    }
    return vec4(red, green, 0, 0.5);
}

void main(){
    xyz_coords = position_point_into_frame(point);
    v_normal = get_rotated_surface_unit_normal_vector(point, du_point, dv_point);
    v_color = red_green_gradient(abs(sin(point.x) * sin(point.y)), 0, 1);
    gl_Position = get_gl_Position(xyz_coords);
}
