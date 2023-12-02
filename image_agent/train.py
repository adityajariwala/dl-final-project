import time

import numpy as np
import torch
import torch.utils.tensorboard as tb
from torchvision import transforms
from sklearn.model_selection import train_test_split

from tournament.utils import load_recording
from .models import save_model, load_model, PuckDetector, UnifiedCoordDist
from .utils import load_detection_data


def train(args):
    from os import path

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(
            args.log_dir, 'train' + '/{}'.format(time.strftime('%H-%M-%S'))), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(
            args.log_dir, 'valid' + '/{}'.format(time.strftime('%H-%M-%S'))), flush_secs=1)

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    pkl_files = ["ai_vs_ai_1200_updated_lowres_with_pucklocation_and_depth_info.pkl",
                 "ai_vs_ai_1800_updated_lowres_with_pucklocation_and_depth_info.pkl"]
    training_data = []
    training_data_with_puck = []

    augmentation = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()]
    )

    for pkl in pkl_files:
        game_data = load_recording(pkl)

        for iteration in game_data:
            for i in range(len(iteration)):
                team1_kart_0_image = iteration[i]['team1_images'][0]
                team1_kart_1_image = iteration[i]['team1_images'][1]
                team2_kart_0_image = iteration[i]['team2_images'][0]
                team2_kart_1_image = iteration[i]['team2_images'][1]
                team1_kart_0_puck_screen_location = iteration[i]['team1_kart_0_puck_screen_location']
                team1_kart_1_puck_screen_location = iteration[i]['team1_kart_1_puck_screen_location']
                team2_kart_0_puck_screen_location = iteration[i]['team2_kart_0_puck_screen_location']
                team2_kart_1_puck_screen_location = iteration[i]['team2_kart_1_puck_screen_location']

                training_data.append((team1_kart_0_image, team1_kart_0_puck_screen_location))
                training_data.append((team1_kart_1_image, team1_kart_1_puck_screen_location))
                training_data.append((team2_kart_0_image, team2_kart_0_puck_screen_location))
                training_data.append((team2_kart_1_image, team2_kart_1_puck_screen_location))

                if team1_kart_0_puck_screen_location[0]:
                    training_data_with_puck.append((team1_kart_0_image, team1_kart_0_puck_screen_location))
                if team1_kart_1_puck_screen_location[0]:
                    training_data_with_puck.append((team1_kart_1_image, team1_kart_1_puck_screen_location))
                if team2_kart_0_puck_screen_location[0]:
                    training_data_with_puck.append((team2_kart_0_image, team2_kart_0_puck_screen_location))
                if team2_kart_1_puck_screen_location[0]:
                    training_data_with_puck.append((team2_kart_1_image, team2_kart_1_puck_screen_location))

    if args.models == "all" or args.models == "puck":
        print("Data loaded, starting training puck binary classifier model...")
        train_data, valid_data = train_test_split(training_data, test_size=0.2)
        train_loader = load_detection_data(train_data, num_workers=4, batch_size=args.batch, transform=augmentation)
        valid_loader = load_detection_data(valid_data, num_workers=4, batch_size=args.batch, transform=augmentation)

        if args.continue_training:
            print("Continuing training model from last saved checkpoint...")
            model_puck = load_model("model_puck.pt").to(device)
        else:
            model_puck = PuckDetector().to(device)

        optimizer_puck = torch.optim.Adam(model_puck.parameters(), lr=args.learning_rate_puck, weight_decay=args.decay)
        loss_puck = torch.nn.CrossEntropyLoss()

        for epoch in range(args.epochs):
            model_puck.train()
            total_loss_puck = 0.
            global_step = 0

            for img, puck, _, _ in train_loader:
                img = img.to(device)
                puck = puck.to(dtype=torch.int).to(device)

                output_puck = model_puck(img)
                loss_val_puck = loss_puck(output_puck, puck).mean()

                optimizer_puck.zero_grad()
                loss_val_puck.backward()
                total_loss_puck += loss_val_puck.detach().cpu().numpy()
                optimizer_puck.step()

                # print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch + 1, args.epochs, global_step + 1,
                #                                                          len(train_loader),
                #                                                          loss_val_puck.detach().cpu().numpy()))

                global_step += 1

            model_puck.eval()
            with torch.no_grad():
                valid_loss = 0
                for img, puck, _, _ in valid_loader:
                    img = img.to(device)
                    puck = puck.to(dtype=torch.int).to(device)
                    output_puck = model_puck(img)
                    val_loss = loss_puck(output_puck, puck).mean()
                    valid_loss += val_loss.detach().cpu().numpy()

            print(f'Epoch {epoch + 1}, Average Validation Loss: {valid_loss}')
            if valid_logger:
                valid_logger.add_scalar('loss', valid_loss, epoch)
            print(f'Epoch {epoch + 1} - avg train loss: {total_loss_puck} - avg valid loss: {valid_loss}')
            save_model(model_puck, f"model_puck_{epoch+1}.pt")
        print()

    elif args.models == "all" or args.models == "unified":
        print("Data loaded, starting training unified model...")
        train_data, valid_data = train_test_split(training_data_with_puck, test_size=0.2)
        train_loader = load_detection_data(train_data, num_workers=4, batch_size=args.batch, transform=augmentation)
        valid_loader = load_detection_data(valid_data, num_workers=4, batch_size=args.batch, transform=augmentation)

        if args.continue_training:
            print("Continuing training model from last saved checkpoint...")
            model_unified = load_model("model_unified.pt").to(device)
        else:
            model_unified = UnifiedCoordDist().to(device)

        optimizer_unified = torch.optim.Adam(model_unified.parameters(), lr=args.learning_rate_unified, weight_decay=args.decay)
        loss_function = torch.nn.SmoothL1Loss()

        for epoch in range(args.epochs):
            model_unified.train()
            global_step = 0
            total_loss_unified = 0.

            for img, _, coord, z in train_loader:
                img = img.to(device)
                coord = coord.to(dtype=torch.float32).to(device)
                z = z.to(dtype=torch.float32).to(device)

                output_coord, output_dist = model_unified(img)

                loss_coord = loss_function(output_coord, coord)
                loss_dist = loss_function(output_dist.squeeze(-1), z)
                total_loss = args.alpha * loss_coord + args.beta * loss_dist

                optimizer_unified.zero_grad()
                total_loss.backward()
                total_loss_unified += total_loss.detach().cpu().numpy()
                optimizer_unified.step()

                # print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch + 1, args.epochs, global_step + 1,
                #                                                          len(train_loader),
                #                                                          total_loss.detach().cpu().numpy()))
                global_step += 1

            total_loss_unified /= len(train_loader)
            if train_logger:
                train_logger.add_scalar('loss', total_loss_unified, epoch)

            model_unified.eval()
            with torch.no_grad():
                valid_loss = 0
                for img, _, coords, z in valid_loader:
                    img, coords, z = img.to(device), coords.to(dtype=torch.float32).to(device), z.to(
                        dtype=torch.float32).to(device)
                    output_coords, output_dist = model_unified(img)
                    val_loss_coord = loss_function(output_coords, coords)
                    val_loss_dist = loss_function(output_dist.squeeze(-1), z)
                    val_total_loss = args.alpha * val_loss_coord + args.beta * val_loss_dist
                    valid_loss += val_total_loss.detach().cpu().numpy()

            valid_loss /= len(valid_loader)
            if valid_logger:
                valid_logger.add_scalar('loss', valid_loss, epoch)
            print(f'Epoch {epoch + 1} - avg train loss: {total_loss_unified} - avg valid loss: {valid_loss}')
            save_model(model_unified, f"model_unified_{epoch+1}.pt")

    if train_logger:
        train_logger.close()
    if valid_logger:
        valid_logger.close()


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-log', '--log_dir', type=str, default='logs')
    # Put custom arguments here
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-t', '--train', type=str, default='data')
    parser.add_argument('-lrp', '--learning_rate_puck', type=float, default=1e-4)
    parser.add_argument('-lru', '--learning_rate_unified', type=float, default=1e-4)
    parser.add_argument('-mo', '--momentum', type=float, default=0.9)
    parser.add_argument('-d', '--decay', type=float, default=1e-4)
    parser.add_argument('-b', '--batch', type=int, default=64)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-m', '--models', type=str, default='all')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    args = parser.parse_args()
    train(args)