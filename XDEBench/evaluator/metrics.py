import piq # [pip install piq]

def normalize_0_1(tensor_gt, tensor_pred):
    min_gt = tensor_gt.reshape(tensor_gt.size(0), -1).min(dim=1)[0]  # [B]
    max_gt = tensor_gt.reshape(tensor_gt.size(0), -1).max(dim=1)[0]  # [B]

    min_gt = min_gt.reshape(-1, 1, 1, 1)
    max_gt = max_gt.reshape(-1, 1, 1, 1)

    min_pred = tensor_pred.reshape(tensor_pred.size(0), -1).min(dim=1)[0]  # [B]
    max_pred = tensor_pred.reshape(tensor_pred.size(0), -1).max(dim=1)[0]  # [B]

    min_pred = min_pred.reshape(-1, 1, 1, 1)
    max_pred = max_pred.reshape(-1, 1, 1, 1)

    epsilon = 1e-8
    norm_tensor_gt = (tensor_gt - min_gt) / (max_gt - min_gt + epsilon)
    norm_tensor_pred = (tensor_pred - min_pred) / (max_pred - min_pred + epsilon)

    return norm_tensor_gt, norm_tensor_pred

def calculate_psnr(tensor_gt, tensor_pred, data_range=1.0):
    """
    Calculate PSNR for grayscale images using piq library
    
    Parameters:
    tensor1: First grayscale image (Tensor)
    tensor2: Second grayscale image (Tensor)
    data_range: Pixel value range (1.0 for [0,1], 255 for [0,255])
    
    Returns:
    PSNR value in dB
    """
    # Validate input format (batch, 1, height, width)
    assert len(tensor_gt.shape) == 4 and tensor_pred.shape[1] == 1, "Input must be 4D grayscale Tensor [B, 1, H, W]"
    assert tensor_gt.shape == tensor_pred.shape, "Input tensors must have identical shapes"

    norm_tensor_gt, norm_tensor_pred = normalize_0_1(tensor_gt, tensor_pred)
    
    return piq.psnr(norm_tensor_gt, norm_tensor_pred, data_range=data_range).item()

def calculate_ssim(tensor_gt, tensor_pred, data_range=1.0):
    """
    Calculate SSIM for grayscale images using piq library
    
    Parameters:
    tensor1: First grayscale image (Tensor)
    tensor2: Second grayscale image (Tensor)
    data_range: Pixel value range (1.0 for [0,1], 255 for [0,255])
    
    Returns:
    SSIM value between 0 and 1
    """
    # Validate input format (batch, 1, height, width)
    assert len(tensor_gt.shape) == 4 and tensor_pred.shape[1] == 1, "Input must be 4D grayscale Tensor [B, 1, H, W]"
    assert tensor_gt.shape == tensor_pred.shape, "Input tensors must have identical shapes"

    norm_tensor_gt, norm_tensor_pred = normalize_0_1(tensor_gt, tensor_pred)

    return piq.ssim(norm_tensor_gt, norm_tensor_pred, data_range=data_range).item()

def calculate_ms_ssim(tensor_gt, tensor_pred, data_range=1.0):
    """
    Calculate multi-scale SSIM for grayscale images
    
    Parameters:
    tensor1: First grayscale image (Tensor)
    tensor2: Second grayscale image (Tensor)
    data_range: Pixel value range (1.0 for [0,1], 255 for [0,255])
    
    Returns:
    MS-SSIM value between 0 and 1
    """
    assert len(tensor_gt.shape) == 4 and tensor_pred.shape[1] == 1, "Input must be 4D grayscale Tensor [B, 1, H, W]"
    assert tensor_gt.shape == tensor_pred.shape, "Input tensors must have identical shapes"

    norm_tensor_gt, norm_tensor_pred = normalize_0_1(tensor_gt, tensor_pred)

    try:
        msssim = piq.multi_scale_ssim(norm_tensor_gt, norm_tensor_pred, data_range=data_range).item()
    except Exception as e:
        try:
            msssim = piq.multi_scale_ssim(norm_tensor_gt, norm_tensor_pred, data_range=data_range, kernel_size=7).item()
        except Exception as e:
            try:
                msssim = piq.multi_scale_ssim(norm_tensor_gt, norm_tensor_pred, data_range=data_range, kernel_size=5).item()
            except Exception as e:
                print(f"Error calculating MS-SSIM with kernel_size=5: {e}")
                msssim = 0.0

    return msssim

def calculate_gmsd(tensor_gt, tensor_pred, data_range=1.0):
    """
    Calculate Gradient Magnitude Similarity Deviation for grayscale images
    
    Parameters:
    tensor1: First grayscale image (Tensor)
    tensor2: Second grayscale image (Tensor)
    data_range: Pixel value range (1.0 for [0,1], 255 for [0,255])
    
    Returns:
    GMSD value between 0 and 1
    """
    assert len(tensor_gt.shape) == 4 and tensor_pred.shape[1] == 1, "Input must be 4D grayscale Tensor [B, 1, H, W]"
    assert tensor_gt.shape == tensor_pred.shape, "Input tensors must have identical shapes"

    norm_tensor_gt, norm_tensor_pred = normalize_0_1(tensor_gt, tensor_pred)

    return piq.gmsd(norm_tensor_gt, norm_tensor_pred, data_range=data_range).item()

def calculate_ms_gmsd(tensor_gt, tensor_pred, data_range=1.0):
    """
    Calculate multi-scale Gradient Magnitude Similarity Deviation for grayscale images
    
    Parameters:
    tensor1: First grayscale image (Tensor)
    tensor2: Second grayscale image (Tensor)
    data_range: Pixel value range (1.0 for [0,1], 255 for [0,255])
    
    Returns:
    MS-GMSD value between 0 and 1
    """
    assert len(tensor_gt.shape) == 4 and tensor_pred.shape[1] == 1, "Input must be 4D grayscale Tensor [B, 1, H, W]"
    assert tensor_gt.shape == tensor_pred.shape, "Input tensors must have identical shapes"

    norm_tensor_gt, norm_tensor_pred = normalize_0_1(tensor_gt, tensor_pred)

    return piq.multi_scale_gmsd(norm_tensor_gt, norm_tensor_pred, data_range=data_range).item()