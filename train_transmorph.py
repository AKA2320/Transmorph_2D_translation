from funcs_transmorph import *
import torch
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

transform = transforms.Compose([
    transforms.ToTensor(),
    BrightestCenterSquareCrop(),
    transforms.Resize((64, 64)),
])

def normalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize a PyTorch tensor to the [0, 1] range.

    Parameters:
    - tensor (torch.Tensor): Input image tensor (any shape, e.g., CxHxW or HxW)

    Returns:
    - torch.Tensor: Normalized tensor with values in [0, 1]
    """
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
        for static, moving in val_loader:
            static = static.to(device).double()
            moving = moving.to(device).double()

            moved_image, pred_translation = model(torch.cat([static, moving], axis=1))
            warped = warper(moving, pred_translation)
            loss = 1 - loss_fn(normalize(warped), normalize(static))
            total_val_loss += loss.item()
    return total_val_loss / len(val_loader)

BATCH_SIZE = 32
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

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    ncc_total_loss = 0

    for static, moving in train_loader:
        static, moving = static.to(DEVICE).double(), moving.to(DEVICE).double()

        moved_image, pred_translation = model(torch.cat([static,moving],axis=1))  # (B, 2)
        warped = warper(moving, pred_translation)

        loss_ncc = ncc_loss_fn(normalize(warped).double(), normalize(static).double())
        # max_range = int(max(warped.max(),static.max()))
        # ssim_loss_fn= SSIM(data_range=1, size_average=True, channel=1).double()
        loss = 1 - ssim_loss_fn(normalize(warped).double(), normalize(static).double())
        # loss = 1 - ms_ssim_loss_fn(normalize(warped).double(), normalize(static).double())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        ncc_total_loss += loss_ncc.item()

    avg_train_loss = total_loss / len(train_loader)
    ncc_avg_loss = ncc_total_loss / len(train_loader)

    avg_val_loss = validate(model, val_loader, ssim_loss_fn, warper, DEVICE)
    # with open('log.txt','a') as f:
    #     f.write(f"Epoch {epoch+1}/{EPOCHS} - NCC Loss: {avg_loss:.4f} \n")
    # print(f"Epoch {epoch+1}/{EPOCHS} - NCC Loss: {avg_loss:.4f}")
    logger.debug(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    logger.debug(f"Loss SSIM Train: {avg_train_loss:.4f}, Loss NCC Train: {ncc_avg_loss:.4f}")




# Assuming 'model' is your trained PyTorch model

# Define the path to save the model
# full_model_save_path = f'model_transmorph_batch{BATCH_SIZE}.pt'

# # Save the entire model
# torch.save(model, full_model_save_path)

# print(f"Full model saved to {full_model_save_path}")