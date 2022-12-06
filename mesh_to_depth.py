import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt


# T_cam_velo = np.array(
#     [
#         [4.27680239e-04, -9.99967248e-01, -8.08449168e-03, -1.19845993e-02],
#         [-7.21062651e-03, 8.08119847e-03, -9.99941316e-01, -5.40398473e-02],
#         [9.99973865e-01, 4.85948581e-04, -7.20693369e-03, -2.92196865e-01],
#         [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
#     ]
# )

# T_velo_cam = np.array(
#     [
#         [4.27679428e-04, -7.21062673e-03, 9.99973959e-01, 2.91804720e-01],
#         [-9.99967209e-01, 8.08119861e-03, 4.85949420e-04, -1.14055066e-02],
#         [-8.08449174e-03, -9.99941381e-01, -7.20693461e-03, -5.6239412e-02],
#         [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
#     ]
# )

#kitti_odometry
T_cam_velo = np.array(
    [
        [-1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03],
        [-6.481465826011e-03, 8.051860151134e-03, -9.999466081774e-01, -7.337429464231e-02],
        [9.999773098287e-01, -1.805528627661e-03, -6.496203536139e-03, -3.339968064433e-01],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)

T_velo_cam = np.array(
    [
        [-0.00185774, -0.00648147,  0.99997723,  0.33350474],
        [-0.99996596,  0.00805186, -0.00180553, -0.00479611],
        [-0.00803997, -0.99994655, -0.0064962 , -0.07557855],
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ]
)

T_gl_cam = np.array(
    [
        [1.0,  0.0,  0.0, 0.0],
        [0.0, -1.0,  0.0, 0.0],
        [0.0,  0.0, -1.0, 0.0],
        [0.0,  0.0,  0.0, 1.0],
    ]
)

T_cam_gl = np.array(
    [
        [1.0,  0.0,  0.0, 0.0],
        [0.0, -1.0,  0.0, 0.0],
        [0.0,  0.0, -1.0, 0.0],
        [0.0,  0.0,  0.0, 1.0],
    ]
)

def cam2vel(poses):
    return T_velo_cam @ poses @ T_cam_velo

def vel2cam(poses):
    return T_cam_velo @ poses @ T_velo_cam

def cam2gl(poses):
    return T_velo_cam @ poses @ T_cam_gl

def render_mesh_to_image(mesh, camera_pose, idx, yfov=np.pi / 3.0, image_width=1226, image_height=370):
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=image_width/image_height)
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                            innerConeAngle=np.pi/16.0,
                            outerConeAngle=np.pi/6.0)
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(image_width, image_height)
    color, depth = r.render(scene)
    print(color.shape)
    print(depth.shape)
    plt.figure()
    plt.subplot(1,2,1)
    plt.axis('off')
    # plt.imshow(color)
    # plt.imsave("color.png", color)
    plt.subplot(1,2,2)
    plt.axis('off')
    # plt.imshow(depth, cmap=plt.cm.gray_r)
    str_idx = '%03d' % idx
    plt.imsave("./data/depth_" + str_idx + ".png", depth, cmap = 'gray')
    # plt.show()

fuze_trimesh = trimesh.load('./data/kitti_odometry_04_depth_10_cropped_p2l_raycasting.ply')
mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
poses_store = np.loadtxt("./data/kitti_odometry_04_depth_10_cropped_p2l_raycasting.txt")
camera_pose = np.array([
    [ 0.0,  0.0, -1.0,  0.0],
    [-1.0,  0.0,  0.0,  0.0],
    [ 0.0,  1.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  1.0],
    ])


idx = 0
for pose in poses_store:
    print(pose)
    pose_4x4 = np.eye(4, 4, dtype=np.float64)
    pose_4x4[:-1] = pose.reshape(3,4)
    pose_4x4 = cam2gl(pose_4x4)
    print(pose_4x4)
    render_mesh_to_image(mesh, pose_4x4, idx, 0.51186)
    idx += 1
