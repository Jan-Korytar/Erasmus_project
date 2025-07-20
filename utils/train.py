import yaml
from torch import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from torchinfo import summary
from transformers import AutoModel, AutoTokenizer

from utils.dataset import TextAndImageDataset
from utils.decoder import Decoder
from utils.helpers import *
from utils.losses import *


def validate(decoder, encoder, tokenizer, dataloader, device):
    decoder.eval()
    encoder.eval()
    criterion = nn.MSELoss(reduction='mean')
    val_loss = 0
    with torch.no_grad():
        for target_image, text, _ in dataloader:
            text_embed = tokenizer(text, return_tensors='pt')
            last_hidden = encoder(**text_embed.to(device)).last_hidden_state
            target_image = target_image.to(device)
            output = decoder(last_hidden)  # predicted image
            loss = criterion(output, target_image)
            val_loss += loss.item()

    val_loss /= len(dataloader)
    return val_loss


def train_decoder(decoder, encoder, tokenizer, train_dataloader, val_dataloader, save_interval=50, num_epochs=100,
                  lr=5e-3, device='cuda'):
    """
    Main training loop, trains the decoder, finetunes the encoder and saves the model.
    :return: train_losses, val_losses
    """
    if torch.cuda.device_count() > 1:
        print(f"--- Using {torch.cuda.device_count()} GPUs ---")
        decoder = nn.DataParallel(decoder)
        encoder = nn.DataParallel(encoder)
    else:
        print(f"--- Using single {device} ---")

    val_losses = []

    train_losses = []
    mae_losses = []
    cl_losses = []
    col_losses = []
    dec_losses = []
    per_losses = []

    decoder = decoder.to(device)
    encoder = encoder.to(device)

    optimizer = torch.optim.AdamW([
        {'params': decoder.parameters(), 'lr': lr},
        {'params': encoder.parameters(), 'lr': lr * 0.1}  # only finetune bert encoder
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=5e-6)

    l_loss = nn.L1Loss(reduction='mean')
    perceptual_loss = PerceptualLoss().to(device)
    clip_loss = CLIPLoss().to(device)
    color_loss = ColorMomentLoss().to(device)
    decorrelation_loss = LatentDecorrelationLoss().to(device)


    decoder.train()
    encoder.train()
    scaler = GradScaler(device)  # needed for mixed precision
    best_loss = float('inf')
    tolerance = 80  # very high, change to have proper early stopping

    for epoch in range(num_epochs):

        epoch_train_loss = 0.0
        epoch_mae = 0.0
        epoch_cl = 0
        epoch_col = 0
        epoch_dec = 0
        epoch_per = 0

        if epoch == 150:
            # no more restarts after epoch 150
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=450, eta_min=5e-6)

        for i, (target_image, text, name) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            with autocast(device):  # memory savings

                tokens = tokenizer(text, return_tensors='pt', max_length=120, truncation=True, padding='max_length')

                last_hidden = encoder(**tokens.to(device)).last_hidden_state  # shape: (B, seq_len, 256)
                target_image: torch.Tensor = target_image.to(device)  # shape: (B, 3, H, W)

                output = decoder(last_hidden)  # predicted image
                latent = decoder.module.latent if isinstance(decoder, nn.DataParallel) else decoder.latent

                mae_loss = 1 * l_loss(output, target_image)
                cl_loss = .5 * clip_loss(output, text)
                col_loss = 0.01 * color_loss(output, target_image)
                dec_loss = 1e-2 * decorrelation_loss(latent)
                per_loss = 0.1 * perceptual_loss(output, target_image)
                loss = mae_loss + cl_loss + dec_loss + col_loss + per_loss

                # + (1e-3 * torch.mean(latent ** 2))) double penalty with adamW

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()
            epoch_mae += mae_loss.item()
            epoch_cl += cl_loss.item()
            epoch_col += col_loss.item()
            epoch_dec += dec_loss.item()
            epoch_per += per_loss.item()

            if i % save_interval == 0 and epoch % 10 == 0:
                names = '_'.join(name)  # save image
                save_sample_images(torch.cat([output[:4], target_image[:4]], dim=0),
                                   filename=f'{epoch:03}_{i}_{names}.jpg')
                plot_train_val_losses(train_losses, val_losses,
                                      {'L1 Loss': mae_losses, 'Color Loss': cl_losses, 'CLIP loss': cl_losses,
                                       'Perceptual Loss': per_losses})

        scheduler.step()
        val_loss = validate(decoder, encoder, tokenizer, val_dataloader, device)
        val_losses.append(val_loss)
        epoch_train_loss /= len(train_dataloader)
        epoch_mae /= len(train_dataloader)
        epoch_cl /= len(train_dataloader)
        epoch_col /= len(train_dataloader)
        epoch_dec /= len(train_dataloader)
        epoch_per /= len(train_dataloader)
        train_losses.append(epoch_train_loss)
        mae_losses.append(epoch_mae)
        cl_losses.append(epoch_cl)
        col_losses.append(epoch_col)
        dec_losses.append(epoch_dec)
        per_losses.append(epoch_per)

        print(
            f"[Epoch {epoch + 1}/{num_epochs}] Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}, Tolerance: {tolerance}")
        print(
            f'MAE: {epoch_mae:.4f}, CLIP: {epoch_cl:.4f}, Color: {epoch_col:.4f}, Decorrelation: {epoch_dec:.4f}, Perceptual: {epoch_per:.4f}')
        if epoch >= 10:
            if val_loss < best_loss:
                tolerance = 35
                best_loss = val_loss
                print(f'Saving the best model at epoch {epoch + 1}/{num_epochs}')
                torch.save(decoder.state_dict(), get_project_root() / 'models' / "decoder_weights.pth")
                torch.save(encoder.state_dict(), get_project_root() / 'models' / "encoder_weights.pth")
            else:
                tolerance -= 1
                if tolerance <= 0:
                    print(f'Early stopping at epoch {epoch + 1}/{num_epochs}')
                    break

    plot_train_val_losses(train_losses, val_losses,
                          {'L1 Loss': mae_losses, 'Color Loss': cl_losses, 'CLIP loss': cl_losses,
                           'Perceptual Loss': per_losses})
    return train_losses, val_losses






if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    project_root = get_project_root()
    with open(project_root / 'config.yml') as f:
        config = yaml.safe_load(f)
        training_config = config['training']
        model_config = config['model']

    tokenizer = AutoTokenizer.from_pretrained(model_config['bert_model'])
    special_tokens_dict = {'additional_special_tokens': ['[NAME]']}
    tokenizer.add_special_tokens(special_tokens_dict)

    bert_encoder = AutoModel.from_pretrained(model_config['bert_model'])
    bert_encoder.resize_token_embeddings(len(tokenizer))
    decoder = Decoder(text_embed_dim=bert_encoder.config.hidden_size, latent_size=model_config['latent_size'],
                      decoder_depth=model_config['decoder_depth'], output_size=model_config['output_size'])


    full_dataset = TextAndImageDataset(project_root / 'data/text_description_concat.csv',
                                       project_root / 'data' / 'images' /
                                       f'{model_config["output_size"][1]}', augment_text=True, augment_images=False)
    # Contrary to the normal train/val/test splits, these are overlapping as pokemons are finite set
    val_dataset = TextAndImageDataset(project_root / 'data/text_description_concat.csv',
                                      project_root / 'data' / 'images' /
                                      f'{model_config["output_size"][1]}', augment_text=False, augment_images=False)

    # Not used code for not overlapping code
    # total_len = len(full_dataset)
    # train_len = int(0.8 * total_len)
    # val_len = int(0.1 * total_len)
    # test_len = len(full_dataset) - train_len - val_len

    # train_dataset, val_dataset, test_dataset = random_split(
    #    full_dataset,
    #    lengths=[train_len, val_len, test_len],
    #    generator=torch.Generator().manual_seed(42)  # for reproducibility
    # )

    val_size = int(len(full_dataset) * 0.1)

    # Generate shuffled indices
    indices = torch.randperm(len(full_dataset)).tolist()[:val_size]
    val_dataset = Subset(val_dataset, indices)


    train_dataloader = DataLoader(full_dataset, batch_size=training_config['batch_size'], shuffle=True,
                                  pin_memory=True, num_workers=1, pin_memory_device=device)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1,
                                pin_memory_device=device)
    if training_config['print_summary']:
        summary(decoder, input_size=(training_config['batch_size'], 120, bert_encoder.config.hidden_size))


    t, v = train_decoder(decoder=decoder, encoder=bert_encoder, tokenizer=tokenizer, train_dataloader=train_dataloader,
                         val_dataloader=val_dataloader, num_epochs=training_config['num_epochs'],
                         lr=float(training_config['learning_rate']), save_interval=training_config['save_interval'],
                         device=device)
