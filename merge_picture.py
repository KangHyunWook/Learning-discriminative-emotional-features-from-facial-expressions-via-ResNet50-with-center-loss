import sys
from PIL import Image

images = [Image.open(x) for x in ['/home/jeff/Documents/centerloss/train_data_no_center_RAF_DB_vis.png',
                                            '/home/jeff/Documents/centerloss/train_data_center_RAF_DB_vis.png']]
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

new_im.save('deep_feats_vis.jpg')


#
