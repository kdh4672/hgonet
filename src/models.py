import os
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import InpaintGenerator, EdgeGenerator, Discriminator, InpaintGenerator_without_edge

from .loss import AdversarialLoss, PerceptualLoss, StyleLoss
from .vit import ViT_Generator
from .vit import Resnet_Generator


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

        model_path = os.path.join(config.PATH,'model')
        try: os.mkdir(model_path)
        except: print(model_path,'already exist')

        self.gen_weights_path_iter = os.path.join(model_path, name )
        self.dis_weights_path_iter = os.path.join(model_path, name)
        self.load_iteration = config.LOAD_ITERATION

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)
            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)
            ##kdkd## config.LOAD이면 state_dict뽑음
            if self.config.LOAD ==True:
                self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def partial_load(self):
        load_path = self.gen_weights_path_iter+'_{}'.format(self.load_iteration)
        print('Partial Loading %s generator...' % self.name)

        if torch.cuda.is_available():
            data = torch.load(load_path + '_gen.pth')
        else:
            data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

        self.generator.load_state_dict(data['generator'])
        self.iteration = data['iteration']
        # load discriminator only when training
        print('Partial Loading %s discriminator...' % self.name)
        load_path = self.dis_weights_path_iter + '_{}'.format(self.load_iteration)
        if torch.cuda.is_available():
            data = torch.load(load_path + '_dis.pth')
        else:
            data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

        self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        ##kdkd-discriminator
        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)

    def partial_save(self):
        print('\nsaving partial weights %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path_iter+'_{}'.format(self.iteration) + '_gen.pth')

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.gen_weights_path_iter+'_{}'.format(self.iteration) + '_dis.pth')


class EdgeModel(BaseModel):
    def __init__(self, config):
        super(EdgeModel, self).__init__('EdgeModel', config)

        # generator input: [grayscale(1) + edge(1) + mask(1)]
        # discriminator input: (grayscale(1) + edge(1))
        # generator = EdgeGenerator(use_spectral_norm=True)
        generator = ViT_Generator(
                image_size_w= 256//2,
                image_size_h= 128,
                patch_size_w=self.config.PATCH_SIZE_W,
                patch_size_h=self.config.PATCH_SIZE_H,
                dim = self.config.DIM,
                depth = self.config.DEPTH,
                heads = self.config.HEADS,
                mlp_dim = self.config.MLP_DIM,
                dropout = self.config.DROPOUT,
                emb_dropout = self.config.EMB_DROPOUT,
                gen_out_pixel = self.config.GEN_PIXEL_SIZE,
                config = self.config
            )
        discriminator = Discriminator(in_channels=2, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)
        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, edges, masks):
        self.iteration += 1


        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = torch.cat((images, edges), dim=1)
        dis_input_fake = torch.cat((images, outputs.detach()), dim=1)
        dis_real, dis_real_feat = self.discriminator(dis_input_real)        # in: (grayscale(1) + edge(1))
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)        # in: (grayscale(1) + edge(1))
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = torch.cat((images, outputs), dim=1)
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)        # in: (grayscale(1) + edge(1))
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
        gen_loss += gen_gan_loss


        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss


        # create logs
        logs = [
            ("l_d1", dis_loss.item()),
            ("l_g1", gen_gan_loss.item()),
            ("l_fm", gen_fm_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks):
        edges_masked = (edges * (1 - masks))
        images_masked = (images * (1 - masks)) + masks
        inputs = torch.cat((images_masked, edges_masked, masks), dim=1)
        outputs = self.generator(inputs)                                    # in: [grayscale(1) + edge(1) + mask(1)]
        return outputs

    # def backward(self, gen_loss=None, dis_loss=None):
    #     if dis_loss is not None:
    #         dis_loss.backward()
    #     self.dis_optimizer.step()

    #     if gen_loss is not None:
    #         gen_loss.backward()
    #     self.gen_optimizer.step()
    def backward(self, gen_loss=None, dis_loss=None):

        if dis_loss is not None:
            dis_loss.backward()
        self.dis_optimizer.step()

        if gen_loss is not None:  # gen_loss first, not the dis_loss
            gen_loss.backward()
        self.gen_optimizer.step()



######################################################################################################
class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)
        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        self.config = config
        if config.WITHOUT_EDGE:
            # generator = InpaintGenerator_without_edge()
            if self.config.RESNET:
                generator = Resnet_Generator(image_size=128)
                pass
            else:
                if self.config.INPUT_SIZE:
                    input_size = self.config.INPUT_SIZE
                    generator = ViT_Generator(
                        # image_size_w= 256//2,
                        # image_size_h= 128,
                        image_size_w= input_size//2,
                        image_size_h= input_size,
                        patch_size_w=self.config.PATCH_SIZE_W,
                        patch_size_h=self.config.PATCH_SIZE_H,
                        dim = self.config.DIM,
                        depth = self.config.DEPTH,
                        heads = self.config.HEADS,
                        mlp_dim = self.config.MLP_DIM,
                        dropout = self.config.DROPOUT,
                        emb_dropout = self.config.EMB_DROPOUT,
                        gen_out_pixel = self.config.GEN_PIXEL_SIZE,
                        config = self.config
                    )
                else: 
                    img_w = 256
                    img_h = 128    
                    generator = ViT_Generator(
                    # image_size_w= 256//2,
                    # image_size_h= 128,
                    image_size_w= img_w//2,
                    image_size_h= img_h,
                    patch_size_w=self.config.PATCH_SIZE_W,
                    patch_size_h=self.config.PATCH_SIZE_H,
                    dim = self.config.DIM,
                    depth = self.config.DEPTH,
                    heads = self.config.HEADS,
                    mlp_dim = self.config.MLP_DIM,
                    dropout = self.config.DROPOUT,
                    emb_dropout = self.config.EMB_DROPOUT,
                    gen_out_pixel = self.config.GEN_PIXEL_SIZE,
                    config = self.config
                )
        else:
            generator = InpaintGenerator()
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
        ##kdkd-discriminator
        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('discriminator', discriminator)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )
        ##kdkd-discriminator
        self.dis_optimizer = optim.Adam(
            params=self.discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, edges, masks):
        self.iteration += 1
        # print(images.shape)
        # zero optimizers
        self.gen_optimizer.zero_grad()
        ##kdkd-discriminator
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs = self(images, edges, masks)
        ##kdkd-discriminator
        # output_total = 
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        ##kdkd-discriminator
        width = images.shape[3]
        

        #kdkd-part-discirminator (나중에 원복해야함) 생성된 이미지중 맨왼쪽만
        # outputs_final = outputs[:,:,:,:self.config.GEN_PIXEL_SIZE]
        # dis_input_real = images[:,:,:,:self.config.GEN_PIXEL_SIZE]
        #kdkd-part-discirminator (나중에 원복해야함)

        #kdkd-bigger_part-discirminator (나중에 원복해야함) 진짜생성된 이미지만
        # outputs_final = outputs
        # dis_input_real = torch.cat((images[:,:,:,:self.config.GEN_PIXEL_SIZE],images[:,:,:,width//2:]),dim=3)
        #kdkd-biiger_part-discirminator (나중에 원복해야함)

        ##kdkd-전체 full size 이미지-discriminator
        outputs_final = torch.cat((outputs[:,:,:,:self.config.GEN_PIXEL_SIZE],images[:,:,:,self.config.GEN_PIXEL_SIZE:width//4],\
            outputs[:,:,:,self.config.GEN_PIXEL_SIZE:-self.config.GEN_PIXEL_SIZE],images[:,:,:,3*width//4:-self.config.GEN_PIXEL_SIZE],outputs[:,:,:,-self.config.GEN_PIXEL_SIZE:]),dim=3)
        dis_input_real = images
        ##kdkd-전체 full size 이미지-discriminator
        # outputs_final = outputs
        dis_input_fake = outputs_final.detach() ##현재 3 128 160이므로 수정해야함
        dis_real, _ = self.discriminator(dis_input_real)                    # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)                    # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        ##kdkd-part-discriminator
        gen_input_fake = outputs_final
        # gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)                    # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        outputs_width = outputs.shape[3]
        # generator l1 loss
        # print("output size {}".format(outputs.shape))
        ##gen_l1_loss = self.l1_loss(outputs, gt) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_l1_loss = self.l1_loss(outputs,torch.cat((images[:,:,:,:self.config.GEN_PIXEL_SIZE],images[:,:,:,width//4:3*width//4],\
            images[:,:,:,-self.config.GEN_PIXEL_SIZE:]),dim=3)) * self.config.L1_LOSS_WEIGHT / (2*self.config.GEN_PIXEL_SIZE/outputs_width)
        gen_loss += gen_l1_loss
    

        # genersator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs_final, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss


        # generator style loss
        gen_style_loss = self.style_loss(outputs_final, images)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss


        # create logs
        logs = [
            # ("l_d2", dis_loss.item()),
            # ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            # ("l_per", gen_content_loss.item()),
            # ("l_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, dis_loss,logs
        # return outputs, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks):
        # images_masked = (images * (1 - masks).float()) + masks
        width = images.shape[3]
        images_masked = images[:,:,:,width//4:3*width//4]
        if self.config.WITHOUT_EDGE:
            inputs = images_masked
        else:
            inputs = torch.cat((images_masked, edges), dim=1)
        # print(inputs.shape)
        outputs = self.generator(inputs)
        # print(outputs.shape)                                    # in: [rgb(3) + edge(1)]
        return outputs

    # def backward(self, gen_loss=None, dis_loss=None):
    #     dis_loss.backward()
    #     self.dis_optimizer.step()

    #     gen_loss.backward()
    #     self.gen_optimizer.step()
    def backward(self, gen_loss=None, dis_loss=None):
        if gen_loss is not None:  # gen_loss first, not the dis_loss
            gen_loss.backward()
        self.gen_optimizer.step()

        if dis_loss is not None:
            dis_loss.backward()
        self.dis_optimizer.step()

#########################################################################
