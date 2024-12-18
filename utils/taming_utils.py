import torch
from PIL import ImageFilter
from gaussian_renderer import render
from .loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
import torchvision.transforms as transforms


def get_edges(image):
    image_pil = transforms.ToPILImage()(image)
    image_gray = image_pil.convert('L')
    image_edges = image_gray.filter(ImageFilter.FIND_EDGES)
    image_edges_tensor = transforms.ToTensor()(image_edges)
    
    return image_edges_tensor

def get_loss_map(reconstructed_image, original_image, config, edges_loss_norm):
    """
    计算加权后的 渲染图像与gt的L1 loss和 gt图像的归一化纹理边缘
        reconstructed_image: 渲染图像, (3,H,W)
        original_image:     gt图像
        config:     score中各项参数的权重，{'view_importance': 50, 'edge_importance': 50, 'mse_importance': 50, 'grad_importance': 25, 'dist_importance': 50, 'opac_importance': 100, 'dept_importance': 5, 'loss_importance': 10, 'radii_importance': 10, 'scale_importance': 25, 'count_importance': 0.1, 'blend_importance': 50}
        edges_loss_norm:    归一化后的 纹理边缘图, (H,W)
    Returns:

    """
    weights = [config["mse_importance"], config["edge_importance"]]
    
    l1_loss = torch.mean(torch.abs(reconstructed_image - original_image), 0).detach()
    l1_loss_norm = (l1_loss - torch.min(l1_loss)) / (torch.max(l1_loss) - torch.min(l1_loss))   # (H,W)

    final_loss = (weights[0] * l1_loss_norm) + \
                (weights[1] * edges_loss_norm)

    return final_loss   # (H,W)

def normalize(config_value, value_tensor):
    multiplier = config_value
    value_tensor[value_tensor.isnan()] = 0

    valid_indices = (value_tensor > 0)
    valid_value = value_tensor[valid_indices].to(torch.float32)

    ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)
    ret_value[valid_indices] = multiplier * (valid_value / torch.median(valid_value))   # 使用中位数进行归一化，并乘以 权重

    return ret_value

def compute_gaussian_score(scene, camlist, edge_losses, gaussians, pipe, bg, importance_values, opt, to_prune=False):
    """
    计算高斯score
        scene:      scenechanging对象
        camlist:    从所有训练相机中 随机选取的x个相机
        edge_losses:    x个相机对应的 纹理边缘图
        gaussians:  gaussians模型对象
        pipe:       渲染管线相关参数
        bg:         背景颜色，默认为 tensor[0, 0, 0]，黑色
        importance_values:  score中各项参数的权重，{'view_importance': 50, 'edge_importance': 50, 'mse_importance': 50, 'grad_importance': 25, 'dist_importance': 50, 'opac_importance': 100, 'dept_importance': 5, 'loss_importance': 10, 'radii_importance': 10, 'scale_importance': 25, 'count_importance': 0.1, 'blend_importance': 50}
        opt:        优化相关参数
        to_prune:   是否剪枝，默认为 False
    """
    config = importance_values

    num_points = len(scene.gaussians.get_xyz)   # 高斯个数 N
    gaussian_importance = torch.zeros((len(camlist), num_points), device="cuda", dtype=torch.float32)   # (x, N)

    all_opacity = scene.gaussians.get_opacity.detach().squeeze()    # 所有高斯的不透明度，(N,)
    all_scales = torch.prod(scene.gaussians.get_scaling.detach(), dim=1)    # 所有高斯的轴长乘积，(N,)

    grads = scene.gaussians.xyz_gradient_accum / scene.gaussians.denom
    grads[grads.isnan()] = 0.0
    all_grads = grads.detach().squeeze()    # 所有高斯中心位置梯度的 均值，(N,)

    for view in range(len(camlist)):    # 遍历随机选取的x个训练视角
        my_viewpoint_cam = camlist[view]
        render_image = render(my_viewpoint_cam, gaussians, pipe, bg)["render"]
        photometric_loss = compute_photometric_loss(my_viewpoint_cam, render_image) # 计算该视角的光度误差，包括L1和SSIM

        gt_image = my_viewpoint_cam.original_image.cuda()
        pixel_weights = get_loss_map(render_image, gt_image, config, edge_losses[view].cuda())  # 计算加权后的 渲染图像与gt的L1 loss和 gt图像的归一化纹理边缘

        render_pkg = render(my_viewpoint_cam, gaussians, pipe, bg, pixel_weights = pixel_weights)

        loss_accum = render_pkg["accum_weights"]
        dist_accum = render_pkg["accum_dist"]   # 每个像素到高斯中心 的距离
        blending_weights = render_pkg["accum_blend"]    # 在渲染当前视角后，每个像素的 blending权重
        reverse_counts = render_pkg["accum_count"]

        visibility_filter = render_pkg["visibility_filter"].detach()

        all_depths = render_pkg["gaussian_depths"].detach()
        all_radii = render_pkg["gaussian_radii"].detach()

        g_importance = (\
            normalize(config["grad_importance"], all_grads) + \
            normalize(config["opac_importance"], all_opacity) + \
            normalize(config["dept_importance"], all_depths) + \
            normalize(config["radii_importance"], all_radii) + \
            normalize(config["scale_importance"], all_scales) )
        
        p_importance = (
                        normalize(config["dist_importance"], dist_accum) + \
                        normalize(config["loss_importance"], loss_accum) + \
                        normalize(config["count_importance"], reverse_counts) + \
                        normalize(config["blend_importance"], blending_weights)
        )

        agg_importance = config["view_importance"] * photometric_loss * (p_importance + g_importance)
        gaussian_importance[view][visibility_filter] = agg_importance[visibility_filter]
    
    gaussian_importance = gaussian_importance.sum(axis = 0)
    return gaussian_importance


def compute_photometric_loss(viewpoint_cam, image):
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
    return loss

def get_count_array(start_count, multiplier, opt, mode):
    """
    根据SfM初始点个数和设定的最终高斯个数 计算 每次增稠后的 高斯个数
        start_count:    SfM结果初始化高斯的个数
        multiplier:     最终的高斯个数的倍率（mode=multiplier） 或 最终的高斯个数（mode=final_count）
        opt:    优化相关参数
        mode:   multiplier模式，高斯最终数量 = SfM数量 * budget(float)；final_count模式，高斯最终数量 = budget(int)
    """
    # Eq. (2) of taming-3dgs
    if mode == "multiplier":
        budget = int(start_count * float(multiplier))
    elif mode == "final_count":
        budget = multiplier
    
    num_steps = ((opt.densify_until_iter - opt.densify_from_iter) // opt.densification_interval)    # 增稠次数
    slope_lower_bound = (budget - start_count) / num_steps  # 每次增稠阶段要 新加的高斯个数

    k = 2 * slope_lower_bound
    a = (budget - start_count - k*num_steps) / (num_steps*num_steps)
    b = k
    c = start_count
    # 每次增稠后 高斯的个数
    values = [int(1*a * (x**2) + (b * x) + c) for x in range(num_steps)]

    return values