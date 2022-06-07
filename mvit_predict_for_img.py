from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from towhee.trainer.utils.visualization import predict_image_classification
# from torchsummary import summary
from torchinfo import summary
import math


from CLS2IDX import CLS2IDX
from config.defaults import get_cfg
from mvit_from_slowfast import MViT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = get_cfg()
cfg_file = '/Users/zilliz/zilliz/vision_transformer_visualization/MVIT_B_16_CONV.yaml'
checkpoint_path = '/Users/zilliz/zilliz/vision_transformer_visualization/checkpoint/IN1K_MVIT_B_16_CONV.pyth'
cfg.merge_from_file(cfg_file)
cfg.TRAIN.ENABLE = False
cfg.TEST.ENABLE = False
cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_path


def print_top_classes(predictions, had_softmaxed=True, **kwargs):
    # Print Top-5 predictions
    if not had_softmaxed:
        prob = torch.softmax(predictions, dim=1)
    else:
        prob = predictions
    class_indices = predictions.data.topk(5, dim=1)[1][0].tolist()
    max_str_len = 0
    class_names = []
    for cls_idx in class_indices:
        class_names.append(CLS2IDX[cls_idx])
        if len(CLS2IDX[cls_idx]) > max_str_len:
            max_str_len = len(CLS2IDX[cls_idx])

    print('Top 5 classes:')
    for cls_idx in class_indices:
        output_string = '\t{} : {}'.format(cls_idx, CLS2IDX[cls_idx])
        output_string += ' ' * (max_str_len - len(CLS2IDX[cls_idx])) + '\t\t'
        output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
        print(output_string)
    return class_indices[0]



# print(cfg)
model = MViT(cfg)
# summary(model, (3, 224, 224))
# summary(model, input_size=(1, 3, 224, 224))

# print(model)
checkpoint = torch.load(checkpoint_path, map_location=device)
# print(checkpoint.keys())
# print(checkpoint['model_state'].keys())
model_state = checkpoint['model_state']

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def change_model_keys(model_state):
    for key in list(model_state.keys()):
        if 'attn.pool' in key or 'attn.norm' in key:
            key_word_list = key.split('.')
            kqv = key_word_list[3][-1]
            norm_pool = key_word_list[3][:4]
            new_key = '.'.join([key_word_list[0], key_word_list[1], 'attn', 'atn_pool_' + kqv, norm_pool, key_word_list[-1]])
            print('old_key: {}\t new_key: {}'.format(key, new_key))
            model_state[new_key] = model_state.pop(key)
    return model_state


model_state = change_model_keys(model_state)
print('-' * 80)
print(model_state.keys())
print('-' * 80)
model.load_state_dict(model_state)
print('load finish')


def vis_heatmap(model, img_path, method='rollout', res_dir = '/Users/zilliz/zilliz/vision_transformer_visualization/res_heatmap'):
    model.eval()
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    image = Image.open(img_path)
    # image = Image.open('my_test_imgs/dog_img.jpeg')
    # image = Image.open('my_test_imgs/bird.jpg')
    img_tensor = transform(image)
    print('img_tensor.shape=', img_tensor.shape)
    predictions = model(img_tensor.to(device).unsqueeze(0))
    top_index = print_top_classes(predictions)
    # prediction_score, pred_label_idx = predict_image_classification(model, input_)
    # output = model(input_)
    # print(prediction_score)
    # print(pred_label_idx)


    # kwargs = {"alpha": 1}
    # if index == None:
    #     index = np.argmax(output.cpu().data.numpy(), axis=-1)
    #
    one_hot = np.zeros((1, predictions.size()[-1]), dtype=np.float32)
    one_hot[0, top_index] = 1
    one_hot_vector = one_hot
    one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(device)
    one_hot = torch.sum(one_hot * predictions).to(device)
    print(one_hot)
    model.zero_grad()
    one_hot.backward(retain_graph=True)
    kwargs = {"alpha": 1}
    # transformer_attribution = model.relprop(torch.tensor(one_hot_vector).to(device), method="rollout", **kwargs)
    # transformer_attribution = model.relprop(torch.tensor(one_hot_vector).to(device), method="transformer_attribution", **kwargs)
    transformer_attribution = model.relprop(torch.tensor(one_hot_vector).to(device), method=method, **kwargs)
    print('transformer_attribution.shape = ', transformer_attribution.shape)

    map_w = int(math.sqrt(transformer_attribution.shape[-1]))
    # transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = transformer_attribution.reshape(1, 1, map_w, map_w)
    scale_factor = 224 // map_w
    print('scale_factor = ', scale_factor)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=scale_factor, mode='bilinear')
    # transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = transformer_attribution.reshape(224, 224).to(device).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
                transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = img_tensor.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
                image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis_img = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(image)
    axs[0].axis('off')

    axs[1].imshow(vis_img)
    axs[1].axis('off')
    # axs[2].imshow(dog)
    axs[2].axis('off')
    # fig.savefig('my_test_res.png', dpi=200)

    # fig.savefig('res_transformer_attribution.png', dpi=200)

    res_path = Path(res_dir) / (str(Path(img_path).name.split('.')[0]) + '_' + method + '.png')
    fig.savefig(res_path, dpi=200)

    # # one_hot = torch.sum(one_hot.cuda() * output)
    #
    # self.model.zero_grad()
    # one_hot.backward(retain_graph=True)
    #
    # return self.model.relprop(torch.tensor(one_hot_vector).to(input.device), method=method, is_ablation=is_ablation,
    #                           start_layer=start_layer, **kwargs)
if __name__ == '__main__':
    img_folder = '/Users/zilliz/zilliz/vision_transformer_visualization/my_test_imgs/n02106166'
    for img_path in Path(img_folder).glob('*.*'):
        vis_heatmap(model, img_path, method='reverse_rollout')