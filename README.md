# Denoising mel-spectrograms with a residual dense network

![Demo](./docs/demo.gif)

## Setup

```bash
# clone the repo
git clone https://github.com/sdll/audio-denoising
cd audio-denoising
docker build --tag audio-denoising:1.0
```

## Run

```bash
docker run -v path_to_dataset:/dataset -v path_to_results:/results --name audio-denoising:1.0
```
