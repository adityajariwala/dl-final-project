import time

import numpy as np
import torch
import torch.utils.tensorboard as tb
from torchvision import transforms

from tournament.utils import load_recording
from .models import save_model, Planner, BinaryClassifier, FoveaNet, load_model, FoveaNetDist
from .utils import load_detection_data


def train(args):
    from os import path
    # model = Detector()
    # model = Planner()

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(
            args.log_dir, 'train' + '/{}'.format(time.strftime('%H-%M-%S'))), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(
            args.log_dir, 'valid' + '/{}'.format(time.strftime('%H-%M-%S'))), flush_secs=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    print(f"Device: {device}")

    # if args.continue_training:
    #     model.load_state_dict(torch.load(
    #         path.join(path.dirname(path.abspath(__file__)), 'det.th')))

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
        train_data = load_detection_data(training_data, num_workers=4, batch_size=args.batch, transform=augmentation)
        if args.continue_training:
            print("Continuing training model from last saved checkpoint...")
            model_puck = load_model("model_puck.pt").to(device)
        else:
            model_puck = BinaryClassifier().to(device)
        optimizer_puck = torch.optim.Adam(model_puck.parameters(), lr=args.learning_rate_puck, weight_decay=1e-5)
        loss_puck = torch.nn.CrossEntropyLoss()

        global_step = 0
        for epoch in range(args.epochs):
            model_puck.train()
            total_loss_puck = 0.

            for img, puck, _, _ in train_data:
                img = img.to(device)
                puck = puck.to(dtype=torch.int).to(device)

                output_puck = model_puck(img)
                loss_val_puck = loss_puck(output_puck, puck).mean()
                total_loss_puck += loss_val_puck.detach().cpu().numpy()

                optimizer_puck.zero_grad()
                loss_val_puck.backward()
                optimizer_puck.step()

                global_step += 1

            model_puck.eval()
            print(f'Epoch: {epoch} - BCLoss: {np.mean(total_loss_puck)}')
            save_model(model_puck, "model_puck.pt")
        print()
    elif args.models == "all" or args.models == "coord":
        print("Data loaded, starting training coord Planner model...")
        train_data_with_puck = load_detection_data(training_data_with_puck, num_workers=4, batch_size=args.batch,
                                                   transform=augmentation)
        if args.continue_training:
            print("Continuing training model from last saved checkpoint...")
            model_coord = load_model("model_coord.pt").to(device)
        else:
            model_coord = FoveaNet().to(device)
        optimizer_coord = torch.optim.Adam(model_coord.parameters(), lr=args.learning_rate_coord)
        loss_coord = torch.nn.SmoothL1Loss()

        global_step = 0
        for epoch in range(args.epochs):
            model_coord.train()
            total_loss_coord = 0.

            for img, _, coord, _ in train_data_with_puck:
                img = img.to(device)
                coord = coord.to(dtype=torch.float32).to(device)

                output_coord = model_coord(img)
                loss_val_coord = loss_coord(output_coord, coord).mean()
                total_loss_coord += loss_val_coord.detach().cpu().numpy()

                optimizer_coord.zero_grad()
                loss_val_coord.backward()
                optimizer_coord.step()

                global_step += 1

            model_coord.eval()
            print(f'Epoch: {epoch} - FNLoss: {np.mean(total_loss_coord)}')
            save_model(model_coord, "model_coord.pt")

    elif args.models == "all" or args.models == "dist":
        print("Data loaded, starting training dist Planner model...")
        train_data_with_puck = load_detection_data(training_data_with_puck, num_workers=4, batch_size=args.batch,
                                                   transform=augmentation)
        if args.continue_training:
            print("Continuing training model from last saved checkpoint...")
            model_dist = load_model("model_dist.pt").to(device)
        else:
            model_dist = FoveaNetDist().to(device)
        optimizer_dist = torch.optim.Adam(model_dist.parameters(), lr=args.learning_rate_dist)
        loss_dist = torch.nn.SmoothL1Loss()

        global_step = 0
        for epoch in range(args.epochs):
            model_dist.train()
            total_loss_dist = 0.

            for img, _, _, z in train_data_with_puck:
                img = img.to(device)
                z = z.to(dtype=torch.float32).to(device)

                output_dist = model_dist(img)
                loss_val_dist = loss_dist(output_dist.squeeze(-1), z).mean()
                total_loss_dist += loss_val_dist.detach().cpu().numpy()

                optimizer_dist.zero_grad()
                loss_val_dist.backward()
                optimizer_dist.step()

                global_step += 1

            model_dist.eval()
            print(f'Epoch: {epoch} - FNDLoss: {np.mean(total_loss_dist)}')
            save_model(model_dist, "model_dist.pt")


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
    parser.add_argument('-e', '--epochs', type=int, default=20)
    parser.add_argument('-t', '--train', type=str, default='data')
    parser.add_argument('-lrp', '--learning_rate_puck', type=float, default=1e-4)
    parser.add_argument('-lrc', '--learning_rate_coord', type=float, default=1e-4)
    parser.add_argument('-lrd', '--learning_rate_dist', type=float, default=1e-4)
    parser.add_argument('-mo', '--momentum', type=float, default=0.9)
    parser.add_argument('-d', '--decay', type=float, default=0.01)
    parser.add_argument('-b', '--batch', type=int, default=50)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-m', '--models', type=str, default='all')
    args = parser.parse_args()
    train(args)