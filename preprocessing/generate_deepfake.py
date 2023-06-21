from os import listdir
from os.path import join, normpath, basename
import random
import yaml
from tqdm.auto import tqdm

import imageio
from skimage.transform import resize
from sync_batchnorm import DataParallelWithCallback
from animate import normalize_kp
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from skimage import img_as_ubyte
import numpy as np
import torch

FOLDER_REAL = "/home/pafvideo/deepaf/dataset-real"
FOLDER_FAKE = "/home/pafvideo/deepaf/dataset-fake"

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.full_load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector

def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


def generateDeepFake(source_image,driving_video,result_video):
    find_best_frame = True
    best_frame = True
    relative = False
    adapt_scale = True
    checkpoint_path = "/home/pafvideo/deepaf/first-order-model-master/vox-cpk.pth.tar"
    config = "/home/pafvideo/deepaf/first-order-model-master/config/vox-256.yaml"

    source_image = imageio.imread(source_image)
    reader = imageio.get_reader(driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    generator, kp_detector = load_checkpoints(config_path=config, checkpoint_path=checkpoint_path, cpu=False)

    if find_best_frame or best_frame is not None:
        i = best_frame if best_frame is not None else find_best_frame(source_image, driving_video, cpu=False)
        print("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector, relative=relative, adapt_movement_scale=adapt_scale, cpu=False)
        predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector, relative=relative, adapt_movement_scale=adapt_scale, cpu=False)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=relative, adapt_movement_scale=adapt_scale, cpu=False)
    imageio.mimsave(result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)



if __name__ == "__main__":
    faces = listdir(join(FOLDER_REAL,"celebfaces"))
    len_faces = len(faces)
    random.shuffle(faces)
    videos = listdir(join(FOLDER_REAL,"train"))
    len_videos = len(videos)
    random.shuffle(videos)
    NB_DEEPFAKES = 25000

    for i in range(NB_DEEPFAKES):
        random_face = join(join(FOLDER_REAL,"celebfaces"),faces[i%len_faces])
        random_video = join(join(FOLDER_REAL,"train"),videos[i%len_videos])
        deepfake_path = join(FOLDER_FAKE,basename(normpath(random_video)))
        #print(random_face, random_video)
        #print("Destination :",join(FOLDER_FAKE,basename(normpath(random_video))))
        generateDeepFake(random_face,random_video,deepfake_path)