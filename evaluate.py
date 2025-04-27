from PATHFinder.utils.dataset_utils import SegmentationDataset
from PATHFinder.utils.visualise import visualize_batch_predictions
from PATHFinder.utils.loss_optim_LR_LR_utils import calculate_eval_metrics
import os
import glob
from torch.utils.data import DataLoader
from PATHFinder.model.decoder import PATHFinder
import argparse

def evaluate_model(model, dataloader, device='cuda'):
    """
        Takes model, dataloader and device as input

        Returns: Binary Predictions of Roads
    
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, annotations in dataloader:
            images = images.to(device)
            annotations = annotations.to(device)

            # Forward pass
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)  # Take the argmax to get predicted classes

            all_preds.append(preds.cpu().numpy())
            all_labels.append(annotations.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


def load_model(checkpoint_path, device='cuda'):
    """
        Loads models into eval state
    """
    model = PATHFinder(num_classes=2)
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model



def main(args):

    # load custom dataset
    model = load_model(args.model_checkpoint, args.device)

    # extract the images and annotations from the provided directory
    # args.imgs_format - e.g tiff |  args.labels_format # e.g png, jpeg,
    # eval_images & eval_labels are lists
    eval_images = sorted(glob.glob(os.path.join(args.images_dir, f"*{args.images_format}")))
    eval_labels = sorted(glob.glob(os.path.join(args.labels_dir, f"*{args.labels_format}")))

    # Dataset and Dataloader
    eval_dataset = SegmentationDataset(images_list = eval_images, 
                        labels_list = eval_labels,
                         
                        num_classes = 2,
                        train=False
                        )
    seg_dataloader = DataLoader(eval_dataset, args.batch_size, shuffle=False)

    # Evaluate model
    predictions, labels = evaluate_model(model, seg_dataloader, device=args.device)

    # Calculate metrics
    miou, mdice = calculate_eval_metrics(predictions, labels, num_classes=2)

    print(f"Mean IoU: {miou:.4f}")
    print(f"Mean Dice: {mdice:.4f}")

    # save predictions and it corresponding labels to directory
    preds_dir = os.path.join(os.getcwd(), 'predicitions_samples')
    os.makedirs(preds_dir, exists_ok=True)    # create save dir
    visualize_batch_predictions(predictions, preds_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(descriptions="Evaluate PATHFinder model.")
    parser.add_argument("--model_checkpoint", required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--images_dir", required=True, help="Directory containing images to evaluate.")
    parser.add_argument("--labels_dir", required=True, help="Directory containing ground truth annotations.")
    parser.add_argument("--images_format", required=True, help="Directory containing ground truth annotations.")
    parser.add_argument("--labels_format", required=True, help="Directory containing ground truth annotations.")
    parser.add_argument("--batch_size", type=int, default=4, help="Directory containing ground truth annotations.")
    parser.add_argument("--device", type=str, default='cuda', help="Device on which computation will run - `cuda` or `cpu`. Default is `cuda`")

    args = parser.parse_args()

    main(args)




