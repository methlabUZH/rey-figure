import numpy as np
import os
import torch


def load_model(checkpoint_fp, model: torch.nn.Module) -> torch.nn.Module:
    if not os.path.isfile(checkpoint_fp):
        raise FileNotFoundError(f'no checkpoint in {checkpoint_fp} found!')

    checkpoint = torch.load(checkpoint_fp, map_location=torch.device('gpu' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    return model


def inference(model: torch.nn.Module, images_numpy: np.ndarray) -> np.ndarray:
    while len(images_numpy.shape) < 4:
        images_numpy = np.expand_dims(images_numpy, axis=0)

    images_tensor = torch.from_numpy(images_numpy).float()
    item_scores, _ = model(images_tensor)

    return item_scores.cpu().numpy()


if __name__ == '__main__':
    from src.models import get_architecture
    from src.training.data_loader import get_dataloader

    #
    # data_root = '/Users/maurice/phd/src/data/psychology/serialized-data/scans-2018-224x224'
    # train_labels_csv = os.path.join(data_root, 'test_labels.csv')
    # data_loader, _ = get_dataloader(data_root=data_root, labels_csv=train_labels_csv,
    #                                 batch_size=2, num_workers=8, shuffle=True,
    #                                 score_type='median', fraction=1.0, mean=None, std=None)

    model = get_architecture('resnet18', num_outputs=18, dropout=None, norm_layer='batch_norm', image_size=[224, 224])
    ckpt = '/Users/maurice/phd/src/psychology/results/sum-score/scans-2018-2021-224x224-augmented/resnet18/2021-09-12_20-03-15.155/checkpoints/model_best.pth.tar'
    checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
    checkpoint['state_dict'] = {str(k).replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    model.eval()

    # for i, (images, labels) in enumerate(data_loader):
    #     with torch.no_grad():
    #         _ = model(images.float())
    #
    #     print(f'completed {i + 1}/30', end='\r', flush=True)
    #
    #     if i > 30:
    #         break

    image_fp = '/Users/maurice/phd/src/data/psychology/serialized-data/scans-2018-224x224/data2018/newupload_9_11_2018/Frederica_1_healthy_050_36.npy'
    image = torch.unsqueeze(torch.from_numpy(np.load(image_fp)), dim=0)
    image = (image - torch.mean(image)) / torch.std(image)
    image = torch.unsqueeze(image, dim=0)

    with torch.no_grad():
        outputs = model(image.float())

    print(outputs)
