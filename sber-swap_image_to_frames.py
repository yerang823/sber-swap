import cv2
import torch
import time
import os
from pathlib import Path

from utils.inference.image_processing import crop_face, get_final_image, show_images
from utils.inference.video_processing import read_video, get_target, get_final_video, add_audio_from_another_video, face_enhancement, save_final_frames
from utils.inference.core import model_inference

from network.AEI_Net import AEI_Net
from coordinate_reg.image_infer import Handler
from insightface_func.face_detect_crop_multi import Face_detect_crop
from arcface_model.iresnet import iresnet100
from models.pix2pix_model import Pix2PixModel
from models.config_sr import TestOptions

from typing import List


def sber_swap(src_image_path: Path, target_video_path: Path, output_dir: Path, aligned=False):
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))

    # main model for generation
    G = AEI_Net(backbone='unet', num_blocks=2, c_id=512)
    G.eval()
    G.load_state_dict(torch.load('weights/G_unet_2blocks.pth', map_location=torch.device('cpu')))
    G = G.cuda()
    G = G.half()

    # arcface model to get face embedding
    netArc = iresnet100(fp16=False)
    netArc.load_state_dict(torch.load('arcface_model/backbone.pth'))
    netArc=netArc.cuda()
    netArc.eval()

    # model to get face landmarks
    handler = Handler('./coordinate_reg/model/2d106det', 0, ctx_id=0, det_size=640)

    # model to make superres of face, set use_sr=True if you want to use super resolution or use_sr=False if you don't
    use_sr = True
    if use_sr:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        torch.backends.cudnn.benchmark = True
        opt = TestOptions()
        #opt.which_epoch ='10_7'
        model = Pix2PixModel(opt)
        model.netG.train()
    
    crop_size = 224 # don't change this
    BS = 60

    source_full = cv2.imread(str(src_image_path))
    source_full = cv2.cvtColor(source_full, cv2.COLOR_BGR2RGB)
    try:
        if not aligned:
            source = crop_face(source_full, app, crop_size)[:1]
        else:
            if source_full.shape[0] > 224:
                source = [cv2.resize(source_full, (224, 224))]
            else:
                source = [source_full]
        print("Everything is ok!")
    except TypeError:
        print("Bad source images")
    
    full_frames, fps = read_video(str(target_video_path))
    target = get_target(full_frames, app, crop_size)

    START_TIME = time.time()

    final_frames_list, crop_frames_list, full_frames, tfm_array_list = model_inference(full_frames,
                                                                                       source,
                                                                                       target,
                                                                                       netArc,
                                                                                       G,
                                                                                       app,
                                                                                       set_target = False,
                                                                                       crop_size=crop_size,
                                                                                       BS=BS)

    if use_sr:
        final_frames_list = face_enhancement(final_frames_list, model)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}_original', exist_ok=True)
    
    save_final_frames(final_frames_list, crop_frames_list, full_frames, tfm_array_list, str(output_dir) , fps, handler)
    for i, frame in enumerate(full_frames):
        frame_idx = f'{i}'.zfill(5)
        
        cv2.imwrite(f"{output_dir}_original/{frame_idx}.jpg", frame)
    print(f"Successfully swap {src_image_path.name} -> {target_video_path.name}")


if __name__ == '__main__':
    src_image = Path('./examples/images/iu5.jpg')
    target_video_path = Path('./examples/videos/random_gif.gif')
    output_dir = Path('/data/jw93/Best_result/tmp/IU_to_random')

    sber_swap(src_image, target_video_path, output_dir)
