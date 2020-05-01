FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime
RUN pip install torchvision pandas numpy tqdm

WORKDIR /usr/src/app
COPY . .

ENV SOURCE_DIR /dataset
ENV TARGET_DIR /results
ENV WEIGHTS ./weights/audio_denoising_psnr_65.1736_epoch_15_D_20_C_6_G_16_G0_16.pth

ENTRYPOINT bash -c "./process.py"


