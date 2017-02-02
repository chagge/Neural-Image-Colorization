import argparse
import generator
import discriminator
from PIL import Image
import train
import tensorflow as tf


#
def parse_args():
    parser = argparse.ArgumentParser(description='Colorize images using conditional generative adversarial networks.')
    parser.add_argument('--input', action='store_false', help='')
    parser.add_argument('--train', action='store_false', help='')
    args = parser.parse_args()

    if args.input:
        colorize_img(args.input)
    elif args.train:
        train_model()
    else:
        parser.print_help()


#
def colorize_img(file_path):
    if not train.Trainer.is_trained():
        print("This model is not yet trained. Rerun colorize.py and invoke --train to resolve this.")
        exit(1)

    filename_queue = tf.train.string_input_producer([file_path])
    reader = tf.WholeFileReader()
    _, data = reader.read(filename_queue)
    in_image = tf.image.decode_jpeg(data, channels=1)

    gen_net = generator.GenerativeNet()
    gen_net.restore('model')
    colorized_image = gen_net.generate(in_image)
    out_image = tf.image.encode_jpeg(colorized_image, format='rgb')

    with tf.Session() as sess:
        img = sess.run(out_image)
        img = Image.fromarray(img, "RGB")
        img.save('')
        print("")


#
def train_model():
    gen_net = generator.GenerativeNet()
    disc_net = discriminator.DiscriminativeNet()
    train.Trainer.train(gen_net, disc_net)


parse_args()
