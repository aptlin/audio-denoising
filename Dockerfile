FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

ENV SOURCE_DIR /dataset
ENV TARGET_DIR /results

WORKDIR /usr/src/app
COPY ./audio_denoising ./weights ./process.py ./

RUN pip install torchvision pandas numpy tqdm

ENTRYPOINT [ "./process.py --weights" ]
CMD [ "./weights/audio_denoising_psnr_59.7472_epoch_6.pth" ]


