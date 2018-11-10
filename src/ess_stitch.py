import numpy as np

def mesh_grid(height,width):
    """Meshgrid in the absolute coordinates.
    :param height: (int) -> image height
    :param width: (int) -> image width
    :Returns:
    (np.array): the linspace grid for the image
    """
    x1_t = np.linspace(-1.0,1.0,width)
    x2_t = np.expand_dims(x1_t, axis=0)
    x_t = (np.ones((height,1)) @ x2_t)
    
    y1_t = np.linspace(-1.0,1.0,height)
    y2_t = np.expand_dims(y1_t,axis=0).T
    y_t = y2_t @ np.ones((1,width))

    x_t = (x_t + 1.0) * 0.5 * (width - 1.0)
    y_t = (y_t + 1.0) * 0.5 * (height - 1.0)

    x_t_flat = np.expand_dims(x_t.flatten(), axis=0)
    y_t_flat = np.expand_dims(y_t.flatten(), axis=0)

    ones = np.ones((x_t_flat))

    grid = np.concatenate((x_t_flat,y_t_flat,ones),0)
    return grid

def cam2world(cam_coords, essential_mat):

    """Convert homogeneous camera coordinates to world coordinates
    :param cam_coords: The camera-based coordinates (transformed from image2cam) -- [B x H * W x 4]
    :param essential_mat: The camera transform matrix -- [4 x 4]
    :return world_coords: the world coordinate matrix -- [B x H * W x 4]
    """
    world_coords = cam_coords @ essential_mat
    return world_coords  # [B x H * W x 4]

def img2cam(depth, pixel_coords, intrinsic_mat_inv):

    """Transform coordinates in the pixel frame to the camera frame.
    :param depth: depth values for the pixels -- [B x H * W]
    :param pixel_coords: Pixel coordinates -- [B x H * W x 3]
    :param intrinsic_mat_inv: The inverse of the intrinsics matrix -- [4 x 4]
    :return:camera_coords: The camera based coords -- [B x H * W x 3]
    """
    print(pixel_coords.shape, intrinsic_mat_inv.shape)
    cam_coords = depth * (pixel_coords @ intrinsic_mat_inv)
    return cam_coords

def world2cam(world_coords, essential_inv):
    """
    :param world_coords: First camera's projected world coordinates -- [B x H * W x 4]
    :param essential_inv: Inverse fundamental matrix -- [4 x 4]
    :return:cam_coords: Camera coordinates from the *other* camera's perspective -- [B x H * W x 4]
    """
    cam_coords = world_coords @ essential_inv
    return cam_coords

def cam2img(cam_coords, intrinsic_mat):

    """Transform coordinates in the camera frame to the pixel frame.
    :param cam_coords: The camera based coords -- [B x H * W x 3]
    :param intrinsic_mat: The camera intrinsics -- [3 x 3]
    :return pixel_coords: The pixel coordinates corresponding to points -- [B x H * W x 2]
    """
    pcoords = cam_coords @ intrinsic_mat  # B x H * W x 3
    x, y, Z = np.split(pcoords, 3, -1)
    seed = 1e-10
    x_norm = x / (Z + seed)  # [B x H * W]
    y_norm = y / (Z + seed)  # [B x H * W]
    return np.concatenate((x_norm, y_norm), -1)  # [B x H * W x 2]

