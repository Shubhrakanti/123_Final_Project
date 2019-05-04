import skvideo.io


video = skvideo.io.vread("simple_shape.png")

print(video.shape)
