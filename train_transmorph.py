from funcs_transmorph import *
import torch
from torchvision import transforms
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CropOrPadToMultiple:
    """
    A PyTorch-compatible transformation to crop or pad a tensor's
    last two dimensions (height, width) to be multiples of m1 and m2.
    It chooses the closest multiple to the original dimension,
    preferring cropping over padding if distances are equal.
    However, if cropping would make a dimension zero (and it wasn't
    originally zero), it will always choose to pad instead.
    It distributes any cropping or padding symmetrically.
    """
    def __init__(self, m1: int, m2: int, pad_value: float = 0.0):
        """
        Initializes the transformation.

        Args:
            m1 (int): The target multiple for the height dimension. Must be a positive integer.
            m2 (int): The target multiple for the width dimension. Must be a positive integer.
            pad_value (float): The value to use for padding. Defaults to 0.0.
        """
        if not (isinstance(m1, int) and m1 > 0 and isinstance(m2, int) and m2 > 0):
            raise ValueError("m1 and m2 must be positive integers.")
        self.m1 = m1
        self.m2 = m2
        self.pad_value = pad_value

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the crop/pad transformation to the input tensor.

        Args:
            x (torch.Tensor): The input tensor. It's expected to have at least
                              two dimensions, with the last two representing
                              height and width (e.g., C, H, W or N, C, H, W).

        Returns:
            torch.Tensor: The modified tensor with height and width dimensions
                          that are exact multiples of m1 and m2.

        Raises:
            ValueError: If the input tensor has fewer than two dimensions.
        """
        if x.ndim < 2:
            raise ValueError("Input tensor must have at least two dimensions (H, W).")

        # Extract original height and width from the last two dimensions
        H, W = x.shape[-2:]

        # --- Determine target height (Hc) ---
        lower_Hc = (H // self.m1) * self.m1
        upper_Hc = ((H + self.m1 - 1) // self.m1) * self.m1

        # Logic: If original dimension is 0, keep it 0.
        # If cropping would make it 0 (lower_Hc == 0) and it wasn't originally 0, then force padding.
        # Otherwise, choose based on closeness, preferring crop if equidistant.
        if H == 0:
            Hc = 0
        elif lower_Hc == 0:
            Hc = upper_Hc
        elif (H - lower_Hc) <= (upper_Hc - H):
            Hc = lower_Hc
        else:
            Hc = upper_Hc

        # --- Determine target width (Wc) ---
        lower_Wc = (W // self.m2) * self.m2
        upper_Wc = ((W + self.m2 - 1) // self.m2) * self.m2

        # Logic: If original dimension is 0, keep it 0.
        # If cropping would make it 0 (lower_Wc == 0) and it wasn't originally 0, then force padding.
        # Otherwise, choose based on closeness, preferring crop if equidistant.
        if W == 0:
            Wc = 0
        elif lower_Wc == 0:
            Wc = upper_Wc
        elif (W - lower_Wc) <= (upper_Wc - W):
            Wc = lower_Wc
        else:
            Wc = upper_Wc

        modified_x = x

        # --- Apply height transformation (crop or pad) ---
        if H > Hc: # Current height is larger than target: need to crop
            crop_amount = H - Hc
            top_crop = crop_amount // 2
            bottom_crop = crop_amount - top_crop
            # Slicing for cropping. Use ... to handle arbitrary leading dimensions.
            modified_x = modified_x[..., top_crop : H - bottom_crop, :]
        elif H < Hc: # Current height is smaller than target: need to pad
            pad_amount = Hc - H
            top_pad = pad_amount // 2
            bottom_pad = pad_amount - top_pad

            # F.pad expects padding from the innermost dimension outwards:
            # (pad_left_W, pad_right_W, pad_top_H, pad_bottom_H, pad_front_C, pad_back_C, ...)
            pad_list_h = []
            # Padding for width (last dimension) - always 0 for height padding
            pad_list_h.extend([0, 0])
            # Padding for height (second to last dimension)
            pad_list_h.extend([top_pad, bottom_pad])
            # Padding for any leading dimensions (e.g., C, N) - always 0
            for _ in range(x.ndim - 2):
                pad_list_h.extend([0, 0])
            pad_dims_h = tuple(pad_list_h)
            modified_x = F.pad(modified_x, pad_dims_h, mode='constant', value=self.pad_value)


        # --- Apply width transformation (crop or pad) ---
        current_W = modified_x.shape[-1] # Get current width after potential height modification

        if current_W > Wc: # Current width is larger than target: need to crop
            crop_amount = current_W - Wc
            left_crop = crop_amount // 2
            right_crop = crop_amount - left_crop
            # Slicing for cropping.
            modified_x = modified_x[..., :, left_crop : current_W - right_crop]
        elif current_W < Wc: # Current width is smaller than target: need to pad
            pad_amount = Wc - current_W
            left_pad = pad_amount // 2
            right_pad = pad_amount - left_pad

            # F.pad expects padding from the innermost dimension outwards:
            # (pad_left_W, pad_right_W, pad_top_H, pad_bottom_H, pad_front_C, pad_back_C, ...)
            pad_list_w = []
            # Padding for width (last dimension)
            pad_list_w.extend([left_pad, right_pad])
            # Padding for height (second to last dimension) and any leading dimensions - always 0
            for _ in range(x.ndim - 1): # For each dimension before W
                pad_list_w.extend([0, 0])
            pad_dims_w = tuple(pad_list_w)
            modified_x = F.pad(modified_x, pad_dims_w, mode='constant', value=self.pad_value)

        return modified_x
    
class CropOrPad():
    """
    Crops or pads a PyTorch image tensor to a given target shape (height, width).
    This transform is compatible with torchvision.transforms.

    If the image is smaller than the target shape in any dimension, it will be
    padded with black pixels (value 0) symmetrically.
    If the image is larger than the target shape in any dimension, it will be
    cropped from the center symmetrically.

    Expects input image tensor to be of shape (C, H, W) or (H, W) for grayscale.
    """
    def __init__(self, target_shape: tuple):
        """
        Args:
            target_shape (tuple): A tuple (target_height, target_width) representing
                                  the desired output shape.
        """
        if not isinstance(target_shape, (tuple, list)) or len(target_shape) != 2:
            raise ValueError("target_shape must be a tuple or list of two integers (height, width).")
        self.target_height, self.target_width = target_shape

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The image cropped or padded to the target shape.
        """
        is_grayscale = False
        if img.dim() == 2: # (H, W) grayscale
            is_grayscale = True
            img = img.unsqueeze(0) # Add a channel dimension: (1, H, W)
        elif img.dim() == 3: # (C, H, W) color
            pass
        else:
            raise ValueError(f"Unsupported image tensor dimensions: {img.dim()}. Expected 2 or 3.")

        current_channels, current_height, current_width = img.shape

        # --- Padding Logic ---
        pad_top = max(0, (self.target_height - current_height) // 2)
        pad_bottom = max(0, self.target_height - current_height - pad_top)
        pad_left = max(0, (self.target_width - current_width) // 2)
        pad_right = max(0, self.target_width - current_width - pad_left)

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            # F.pad expects padding in the order (left, right, top, bottom) for 2D spatial dims
            img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

        # --- Cropping Logic ---
        # Recalculate dimensions after potential padding
        _, current_height_padded, current_width_padded = img.shape

        if current_height_padded > self.target_height or current_width_padded > self.target_width:
            crop_start_h = max(0, (current_height_padded - self.target_height) // 2)
            crop_end_h = crop_start_h + self.target_height
            crop_start_w = max(0, (current_width_padded - self.target_width) // 2)
            crop_end_w = crop_start_w + self.target_width

            # Crop the image
            img = img[:, crop_start_h:crop_end_h, crop_start_w:crop_end_w]

        if is_grayscale:
            img = img.squeeze(0) # Remove the channel dimension if it was grayscale initially

        return img


transform = transforms.Compose([
    transforms.ToTensor(),
    CropOrPad((64,416)),
    # BrightestCenterSquareCrop(),
    # transforms.Resize((64, 64)),
])

def normalize(tensor: torch.Tensor) -> torch.Tensor:
    min_val = tensor.min()
    max_val = tensor.max()

    # Prevent division by zero if all values are the same
    if max_val == min_val:
        return torch.zeros_like(tensor)

    return (tensor - min_val) / (max_val - min_val)

def validate(model, val_loader, loss_fn, warper, device):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for static, moving, shift_vals in val_loader:
            static = normalize(static.to(device)).double()
            moving = normalize(moving.to(device)).double()
            shift_vals = shift_vals.to(DEVICE).double()

            moved_image, pred_translation = model(torch.cat([static, moving], axis=1))
            warped = warper(moving, pred_translation)
            loss_ncc = 1 - loss_fn(normalize(warped).double(), normalize(static).double())
            loss_trans = F.mse_loss(shift_vals, pred_translation)
            weighted_loss = ncc_loss_weight * loss_ncc + trans_loss_weight * loss_trans

            total_val_loss += weighted_loss.item()
    return total_val_loss / len(val_loader)

BATCH_SIZE = 16
train_dataset = imagepairdataset(root_dir='train', transform = transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = imagepairdataset(root_dir='val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
EPOCHS = 150
print('USING: ',DEVICE)
model = TransMorph(CONFIGS['TransMorph']).to(DEVICE).double()
ncc_loss_fn = NCCLoss().double()
ssim_loss_fn= SSIM(data_range=1, size_average=True, channel=1).double()
ms_ssim_loss_fn= MS_SSIM(data_range=1, size_average=True, channel=1).double()
warper = SpatialTransformer(size=(2, 2)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
ncc_loss_weight = 0.3
trans_loss_weight = 0.7

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    ncc_total_loss = 0

    for static, moving, shift_vals in train_loader:
        static, moving = normalize(static.to(DEVICE).double()), normalize(moving.to(DEVICE).double())
        shift_vals = shift_vals.to(DEVICE).double()

        moved_image, pred_translation = model(torch.cat([static,moving],axis=1))  # (B, 2)
        warped = warper(moving, pred_translation)

        loss_ncc = 1 - ncc_loss_fn(normalize(warped).double(), normalize(static).double())
        loss_trans = F.mse_loss(shift_vals, pred_translation)
        # print(loss_ncc, loss_trans)
        weighted_loss = ncc_loss_weight * loss_ncc + trans_loss_weight * loss_trans
        # max_range = int(max(warped.max(),static.max()))
        # ssim_loss_fn= SSIM(data_range=1, size_average=True, channel=1).double()
        # loss = 1 - ssim_loss_fn(normalize(warped).double(), normalize(static).double())
        # loss = 1 - ms_ssim_loss_fn(normalize(warped).double(), normalize(static).double())

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        total_loss += weighted_loss.item()
        # ncc_total_loss += loss_ncc.item()
        break

    avg_train_loss = total_loss / len(train_loader)
    # ncc_avg_loss = ncc_total_loss / len(train_loader)

    avg_val_loss = validate(model, val_loader, ncc_loss_fn, warper, DEVICE)
    # with open('log.txt','a') as f:
    #     f.write(f"Epoch {epoch+1}/{EPOCHS} - NCC Loss: {avg_loss:.4f} \n")
    # print(f"Epoch {epoch+1}/{EPOCHS} - NCC Loss: {avg_loss:.4f}")
    logger.debug(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    break
    # logger.debug(f"Loss SSIM Train: {avg_train_loss:.4f}, Loss NCC Train: {ncc_avg_loss:.4f}")




# Assuming 'model' is your trained PyTorch model

# Define the path to save the model
# full_model_save_path = f'model_transmorph_batch{BATCH_SIZE}_ncc_normalized_shiftrange3_dynamiccrop.pt'

# # Save the entire model
# torch.save(model, full_model_save_path)

# print(f"Full model saved to {full_model_save_path}")