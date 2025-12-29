from funcs_transmorph import *
import torch
from torchvision import transforms
from pytorch_msssim import SSIM, MS_SSIM
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CropOrPad():
    def __init__(self, target_shape: tuple):
        if not isinstance(target_shape, (tuple, list)) or len(target_shape) != 2:
            raise ValueError("target_shape must be a tuple or list of two integers (height, width).")
        self.target_height, self.target_width = target_shape

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
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
            img = img.squeeze(0)

        return img


transform = transforms.Compose([
    transforms.ToTensor(),
    CropOrPad((64,416)),
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


full_model_save_path = f'model_transmorph_batch{BATCH_SIZE}_ncc_normalized_shiftrange3_dynamiccrop.pt'

# # Save the entire model
# torch.save(model, full_model_save_path)

# print(f"Full model saved to {full_model_save_path}")


# # Use torch.jit.script to convert the model
example_input = torch.randn(1, 2, 64, 416, dtype=torch.double, device=DEVICE) 

# 2. Trace the Model
# Pass the model and the example input to torch.jit.trace
traced_model = torch.jit.trace(model, example_input)
traced_model.save(full_model_save_path)