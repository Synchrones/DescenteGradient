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

vec4 linearize_color(vec4 color){
    vec4 new_color = color;
    for(int i = 0; i < 3; ++i){
        if(new_color[i] <= 0.04045){
            new_color[i] = new_color[i] / 12.92;
        }
        else{
            new_color[i] = pow(((new_color[i] + 0.055) / 1.055), 2.4);
        }
    }
    return new_color;
}

vec4 unlinearize_color(vec4 color){
    vec4 new_color = color;
    for(int i = 0; i < 3; ++i){
        if (new_color[i] <= 0.0031308){
            new_color[i] = 12.92 * new_color[i];
        }
        else{
            new_color[i] = (1.055 * pow(new_color[i], (1/2.4))) - 0.055;
        }
    }
    return new_color;
}


vec4 red_green_gradient(float value, float min_value, float max_value){
    vec4 red_lin = linearize_color(vec4(1, 0, 0, 1));
    vec4 green_lin = linearize_color(vec4(0, 1, 0, 1));

    float lerp_value = 1 - (value - min_value) / (max_value - min_value);
    vec4 lerped_lin_color = vec4(red_lin[0] * (1-lerp_value) + green_lin[0] * lerp_value, red_lin[1] * (1-lerp_value) + green_lin[1] * lerp_value, 0, 0.9);
    return unlinearize_color(lerped_lin_color);
}

void main(){
    xyz_coords = position_point_into_frame(point);
    v_normal = get_rotated_surface_unit_normal_vector(point, du_point, dv_point);
    v_color = red_green_gradient(abs(cos(point.x) * cos(point.y)), 0, 1);
    gl_Position = get_gl_Position(xyz_coords);
}
