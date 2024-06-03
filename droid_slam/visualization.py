import torch
import cv2
import lietorch
import droid_backends
import numpy as np
import open3d as o3d
from lietorch import SE3
import geom.projective_ops as pops
import matplotlib.pyplot as plt
import logging

CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def create_camera_actor(g, scale=0.05):
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))
    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor

def create_point_actor(points, colors):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

def droid_visualization(video, device="cuda:0"):
    torch.cuda.set_device(device)
    video.cameras = {}
    video.points = {}
    warmup = 8
    scale = 1.0
    ix = 0
    filter_thresh = 0.005

    poses = video.poses.cpu()
    disps = video.disps.cpu()
    images = video.images.cpu()[:, [2,1,0], 3::8, 3::8].permute(0, 2, 3, 1) / 255.0
    intrinsics = video.intrinsics[0].cpu()

    # OpenCV video writer setup
    height, width = 540, 960
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    for t in range(video.counter.value):
        pose = SE3(poses[t]).inv().matrix().cpu().numpy()
        disp = disps[t]
        image = images[t]
        logging.error(SE3(poses[t]).inv().data.shape)
        logging.error(SE3(poses[t]).inv().data)

        points = droid_backends.iproj(SE3(poses[t]).inv().data, disp, intrinsics).cpu()

        count = droid_backends.depth_filter(
            poses, disps, intrinsics, torch.tensor([t]), torch.tensor([filter_thresh]))

        count = count.cpu()
        mask = ((count >= 2) & (disp > .5 * disp.mean(dim=[1, 2], keepdim=True))).reshape(-1)
        pts = points.reshape(-1, 3)[mask].numpy()
        clr = image.reshape(-1, 3)[mask].numpy()

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        cam_actor = create_camera_actor(True)
        cam_actor.transform(pose)
        vis.add_geometry(cam_actor)

        point_actor = create_point_actor(pts, clr)
        vis.add_geometry(point_actor)

        vis.poll_events()
        vis.update_renderer()

        img = vis.capture_screen_float_buffer(False)
        vis.destroy_window()

        img = (np.asarray(img) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Write frame to video
        out.write(img)

    out.release()
    print("Video saved as output.mp4")

# 例としての呼び出し方法
# video = ...  # DROIDシステムから得られたビデオデータ
# droid_visualization(video)
