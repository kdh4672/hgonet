import os
def check_link(in_dir, basename, out_dir):
    in_file = os.path.join(in_dir, basename)
    if os.path.exists(in_file):
        link_file = os.path.join(out_dir, basename)
        rel_link = os.path.relpath(in_file, out_dir)  # from out_dir to in_file
        os.symlink(rel_link, link_file)
 
def add_splits(data_path):
    images_path = os.path.join(data_path, 'img_align_celeba')
    train_dir = os.path.join(data_path, 'celeba', 'train')
    valid_dir = os.path.join(data_path, 'celeba', 'valid')
    test_dir = os.path.join(data_path, 'celeba', 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
 
    # these constants based on the standard CelebA splits
    NUM_EXAMPLES = 202599
    TRAIN_STOP = 162770
    VALID_STOP = 182637
 
    for i in range(0, TRAIN_STOP):
        basename = "{:06d}.jpg".format(i+1)
        check_link(images_path, basename, train_dir)
    for i in range(TRAIN_STOP, VALID_STOP):
        basename = "{:06d}.jpg".format(i+1)
        check_link(images_path, basename, valid_dir)
    for i in range(VALID_STOP, NUM_EXAMPLES):
        basename = "{:06d}.jpg".format(i+1)
        check_link(images_path, basename, test_dir)
 
if __name__ == '__main__':
    base_path = '/home/ubuntu/database/'
    add_splits(base_path)