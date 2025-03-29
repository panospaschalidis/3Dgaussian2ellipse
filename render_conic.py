import moderngl
import numpy as np
import torch
import pdb
import matplotlib.pyplot as plt
import visdom

from torchvision.utils import save_image
from PIL import Image


def render(centers, colors, scales, rotations, cam):
    H = cam['image_height']
    W = cam['image_width']

    ctx = moderngl.create_context(standalone=True)
    prog = ctx.program(
        vertex_shader="""
            #version 330

            in vec2 in_vertex;
            in vec2 in_center;
            in vec4 in_color;
            in float in_radius_1;
            in float in_radius_2;
            out vec2 v_vertex;
            out vec2 v_center;
            out vec4 v_color;
            out float v_radius_1;
            out float v_radius_2;
            out vec4 out_position;

            void main() {
                v_vertex = in_vertex;
                v_center = in_center;
                v_color = in_color;
                v_radius_1 = in_radius_1;
                v_radius_2 = in_radius_2;
                out_position = vec4(in_vertex, 0.0, 1.0);
                gl_Position = out_position;
            }
        """,
        fragment_shader="""
            #version 330

            in vec2 v_vertex;
            in vec2 v_center;
            in vec4 v_color;
            in float v_radius_1;
            in float v_radius_2;
            out vec4 f_color;
            
            void main() {
                float d = sqrt(dot(v_vertex-v_center, v_vertex-v_center));
                float r1 = v_radius_1*v_radius_2;
                vec2 vector = normalize((v_vertex - v_center));
                vec2 ortho_xy = vec2(0,1);
                float sin_theta = dot(vector, ortho_xy);
                float sin_theta2 = sin_theta*sin_theta;
                float cos_theta2 = 1-sin_theta2;
                float a2 = v_radius_1*v_radius_1;
                float b2 = v_radius_2*v_radius_2;
                float r2 = sqrt(a2*sin_theta2 + b2*cos_theta2);
                if (d > (r1/r2)) {
                    discard;
                }
                float v_radius = r1/r2;
                f_color = vec4(v_color.xyz, v_color.w);
            } 
        """,
        varyings=['out_position']
    )
    #point = centers[13721]#np.array([[9.4/1920, 9.8/1080]])
    #pdb.set_trace()
    rectangle = np.array([[0,0],
                     [3,0],
                     [0,1],
                     [3,1]]).astype(np.float32)
    center = rectangle.mean(axis=0)
    normal = rectangle[1]-rectangle[3]
    normal /= np.sqrt(np.dot(normal, normal))
    max_radius_1 = np.abs(np.dot(normal, rectangle[3]-center))

    normal = rectangle[2]-rectangle[3]
    normal /= np.sqrt(np.dot(normal, normal))
    max_radius_2 = np.abs(np.dot(normal, rectangle[2]-center))
    rectangle -= center
    rectangle[:,0] = rectangle[:,0]/max_radius_2
    rectangle[:,1] = rectangle[:,1]/max_radius_1
    rectangle_vertices = rectangle[
                        [
                        0,1,3,
                        0,2,3
                        ]
                        ][np.newaxis]
    #pdb.set_trace()
    vertices = (rotations[:,None,:,:] @ rectangle_vertices[...,None])[...,-1]
    vertices = scales[:,np.newaxis] * vertices  
    vertices += centers[:,np.newaxis]
    vertices = vertices.reshape(-1, 2)
    center_ = np.repeat(centers, 6, axis=0)
    color = np.repeat(colors, 6, axis=0)
    radii_1 = np.repeat(scales[:,0][:, np.newaxis], 6, axis=0)
    radii_2 = np.repeat(scales[:,1][:, np.newaxis], 6, axis=0)
    data = np.hstack([vertices, center_, color, radii_1, radii_2]).astype(np.float32)

    #pdb.set_trace()
    vbo = ctx.buffer(data.tobytes())

    vao = ctx.vertex_array(prog, vbo, "in_vertex", "in_center", "in_color", "in_radius_1", "in_radius_2")


    light = np.array([-0.5, -0.8, -2], dtype=np.float32)



    #prog['light'].write(light.tobytes())
    
    fbo = ctx.framebuffer(
        color_attachments=[ctx.texture((1920, 1080), 3)]
    )
    feedback_buffer = ctx.buffer(reserve=vertices.nbytes)
    feedback_vao = ctx.vertex_array(prog, vbo,  "in_vertex","in_center", "in_color", "in_radius_1","in_radius_2")
    feedback_vao.transform(feedback_buffer, moderngl.POINTS)
    transformed_positions = np.frombuffer(feedback_buffer.read(), dtype='f4')
    print(transformed_positions)
    fbo.use()
    fbo.clear(1.0, 1.0, 1.0, 1.0)
    vao.render()

    I = Image.frombytes(
        "RGB", fbo.size, fbo.color_attachments[0].read(),
        "raw", "RGB", 0, -1
    )
    
    plt.imshow(np.array(I)[::-1])
    #plt.axis([-2000,4000,-500,2500])
    #transformed_positions = transformed_positions.reshape(len(transformed_positions)//2, 2)
    #points = transformed_positions.copy()
    #points[:,0] = points[:,0]*960+960
    #points[:,1] = points[:,1]*540+540
    #plt.plot(points[:,0], points[:,1], 'r*')
    #plt.plot(center_[0,0]*960+960, center_[0,1]*540+540, 'mo')
    #plt.plot(centers[:,0]*(1920-1), centers[:,1]*(1080-1), 'r*')
    #plt.show()
    #Image.fromarray(np.array(I)[::-1])
    vis = visdom.Visdom()
    #vis.image(torch.tensor(np.array(I)[::-1]).permute(2,0,1))
    vis.image(np.array(I)[::-1].transpose(2,0,1))
    pdb.set_trace()
    #print(np.unique(transformed_positions, axis=0))

